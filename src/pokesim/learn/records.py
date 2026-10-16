"""Actor views and chosen actions from our own ladder records, for learning the opponent's and our policy.

  uv run python -m pokesim.learn.records --split train --out data/processed/learn/ladder-train-v1.jsonl.gz \
      [--ledger data/curated/v2/battles.jsonl --manifest data/curated/split_manifest_v2.json]

Every normal turn of every record in the split becomes two rows: what we (p1) saw and did, and what the opponent
(p2) saw and did. A view is exactly what the live search's public policies receive (services/sim-tools/observe.js:
the side's own request plus the public state and history up to the start of the turn); nothing later in the game
is used. The opponent's own request holds a set sampled from the training prior that contains the move it played
(or the Pokémon it switched to), the way the search's hypothesis worlds hold a full set for it. Our rows also
carry the search's root evidence as a teacher for distillation.

Rows: schema_version, source, battle_id, record, started, turn, side, split, view, legal, action, outcome
(1 win / 0 loss / .5 draw for that side), rating {p1, p2}, opponent, teacher (p1 only), sample_weight.
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import time
from pathlib import Path

from pokesim.config import FORMAT, ROOT
from pokesim.data.curate_battles import CURATED
from pokesim.data.dataset import DatasetScope, write_metadata
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id
from pokesim.search.lookahead import Lookahead, field_state, mon_state, revealed, side_conditions

SCHEMA = 1
PRIORS = ROOT / 'data/processed/scoped/priors.json'


def ratings(log: list[str]) -> dict[str, int | None]:
    out = {'p1': None, 'p2': None}
    for line in log:
        parts = line.split('|')
        if len(parts) > 5 and parts[1] == 'player' and parts[2] in out and parts[5].isdigit():
            out[parts[2]] = int(parts[5])
    return out


def turn_actions(log: list[str], start: int, stones: dict | None = None) -> dict[str, str | None]:
    """Each side's turn decision after log[start] (a |turn| line): its first move or switch before the next turn,
    a Mega Evolution announced before the move marking it. None when that side made no visible choice.
    stones (optional dict): filled with the Mega Stone each side revealed this turn."""
    out: dict[str, str | None] = {'p1': None, 'p2': None}
    megas: set[str] = set()
    for line in log[start + 1:]:
        parts = line.split('|')
        if len(parts) < 2:
            continue
        kind = parts[1]
        if kind in ('turn', 'win', 'tie'):
            break
        if len(parts) < 3 or parts[2][:2] not in out:
            continue
        side = parts[2][:2]
        if kind == '-mega':
            megas.add(side)
            if stones is not None and len(parts) > 4:
                stones[side] = to_id(parts[4])
        if kind in ('cant', 'faint') and out[side] is None:
            out[side] = ''  # no choice this turn (asleep, flinched, fainted before acting)
        if kind in ('move', 'switch') and '[from]' not in line and not line.endswith('[still]') and out[side] is None:
            if kind == 'move':
                out[side] = 'move:' + to_id(parts[3]) + (':mega' if side in megas else '')
            else:
                out[side] = 'switch:' + to_id(parts[3].split(',')[0])
    return {k: (v or None) for k, v in out.items()}


def opponent_world(engine: Lookahead, battle, action: str | None, stone: str | None = None) -> dict:
    """The opponent's side as the search samples it, with the action it played made possible: its move among the
    active's moves (its Mega Stone when it Mega Evolved), or the Pokémon it switched to among the sampled bench."""
    active = battle.opponent_active_pokemon
    seen = [active] + [m for m in battle.opponent_team.values() if m is not active]
    seen_ids = {engine.sampler.base_id(m.species) for m in seen}
    unseen = [m.species for m in battle.teampreview_opponent_team if engine.sampler.base_id(m.species) not in seen_ids]
    engine.sampler.rng.shuffle(unseen)
    if action and action.startswith('switch:'):
        target = action.split(':')[1]
        for species in unseen:
            if engine.sampler.base_id(to_id(species)) == target:
                unseen.remove(species)
                unseen.insert(0, species)
    team, mons = [], []
    for i, m in enumerate(seen[:3]):
        moves, item, ability = revealed(m)
        moves = list(moves)
        if i == 0 and action and action.startswith('move:'):
            move = action.split(':')[1]
            if move not in moves:
                moves.append(move)
            if action.endswith(':mega') and stone:
                item = stone
        team.append(engine.sampler.sample(m.species, moves, item, ability))
        mons.append(mon_state(m, i == 0))
        if i == 0 and hasattr(battle, 'state_tracker'):
            mons[-1].update(battle.state_tracker.locks(battle.opponent_role, m.name))
    for sid in unseen[:3 - len(team)]:
        team.append(engine.sampler.sample(sid))
        mons.append({'hp': 1.0})
    return {'team': team, 'mons': mons, 'megaUsed': battle.opponent_used_mega_evolve,
            'conditions': side_conditions(battle.opponent_side_conditions, battle.turn)}


def teacher_of(rec: dict, turn: int) -> dict | None:
    for entry in rec.get('advice') or []:
        if entry.get('turn') == turn and not entry.get('force') and entry.get('chosen'):
            status = (entry.get('status') or {}).get('search') or {}
            if status.get('root_evidence'):
                rho = ((rec.get('versions') or {}).get('planner') or {}).get('rho', .3)
                return {'chosen': entry['chosen'], 'root_evidence': status['root_evidence'], 'rho': rho,
                        'fallback': bool((entry.get('status') or {}).get('fallback')),
                        'failed': status.get('terminal_search', {}).get('failed', 0)}
    return None


def swap_roles(lines: list[str]) -> list[str]:
    """The protocol seen with p1 and p2 exchanged (the worker always takes the actor as p1)."""
    return [re.sub(r'\bp([12])(?=[a-z]?[:|]|$)', lambda m: 'p' + ('2' if m[1] == '1' else '1'), line)
            for line in lines]


def rows_from_session(battle_id: str, session: BattleSession, lines: list[str], role: str, engine: Lookahead,
                      own: dict, meta: dict, sides=('p1', 'p2'), teacher=None) -> tuple[list[dict], dict]:
    """Rows of every normal turn: `lines` are applied to `session` (the actor plays `role`), and at each |turn|
    the actor's view ('p1' in `sides`) and the foe's view ('p2') are taken from one sampled world.
    meta: fields copied into every row (source, record, started, outcome for the actor, rating, opponent).
    teacher(turn) -> dict | None: the actor's search evidence at that turn (ladder records)."""
    foe = 'p2' if role == 'p1' else 'p1'
    rows, skipped = [], {'no_choice': 0, 'unreplayable': 0, 'illegal': 0}
    turns = sum(line.startswith('|turn|') for line in lines)
    for i, line in enumerate(lines):
        if line.startswith('|error|'):
            continue
        try:
            session.apply([line])
        except Exception:  # noqa: BLE001 - a log the session cannot follow ends the usable prefix
            skipped['unreplayable'] += 1
            break
        if not line.startswith('|turn|'):
            continue
        turn = int(line.split('|')[2])
        stones: dict = {}
        actions = turn_actions(lines, i, stones)
        try:
            battle = session.decision()
            if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
                skipped['no_choice'] += 1
                continue
            engine.sampler.rng = random.Random(int(hashlib.sha256(f'{battle_id}:{turn}'.encode()).hexdigest()[:12],
                                                   16))
            world = {'p1': engine.own_side(battle, own),
                     'p2': opponent_world(engine, battle, actions[foe], stones.get(foe)),
                     'field': field_state(battle)}
            history = list(session.log) if role == 'p1' else swap_roles(session.log)
            reply = engine._query({'observe': True, 'format': FORMAT, 'samples': [world],
                                   'observationLog': history, 'sides': list(sides)})
        except Exception as e:  # noqa: BLE001 - one turn we cannot rebuild must not drop the battle
            skipped['unreplayable'] += 1
            skipped.setdefault('errors', []).append(f'T{turn}: {e!r}'[:160])
            continue
        views = reply['observations'][0]
        for side in sides:
            action = actions[role if side == 'p1' else foe]
            if not action:
                skipped['no_choice'] += 1
                continue
            legal = [label for _, label in views[side]['legal']]
            if action not in legal:
                skipped['illegal'] += 1
                continue
            row = {'schema_version': SCHEMA, **meta, 'battle_id': battle_id, 'turn': turn, 'turns': turns,
                   'side': side, 'view': views[side]['view'], 'legal': legal, 'action': action,
                   'outcome': meta['outcome'] if side == 'p1' else 1 - meta['outcome'], 'sample_weight': 1.0}
            if side == 'p1' and teacher is not None:
                row['teacher'] = teacher(turn)
            rows.append(row)
    return rows, skipped


def rows_of(battle_id: str, rec: dict, sets: list[dict], engine: Lookahead, own: dict, split: str,
            sides=('p1', 'p2')) -> tuple[list[dict], dict]:
    """All normal-turn rows of one of our records (us = p1), and counts of what was skipped."""
    log = rec['log']
    if rec.get('result') not in ('win', 'loss'):
        return [], {'no_result': 1}
    session = BattleSession(sets)
    session.preview(rec['preview']['species'])
    session.select([next(i for i, s in enumerate(sets) if to_id(s['species']) == to_id(x)) for x in rec['lineup']])
    start = next(i for i, line in enumerate(log) if line.startswith('|teampreview')) + 1
    meta = {'source': 'ladder', 'record': rec.get('source'), 'started': rec.get('started'), 'split': split,
            'outcome': {'win': 1., 'loss': 0.}.get(rec.get('result'), .5), 'rating': ratings(log),
            'opponent': rec.get('opponent')}
    return rows_from_session(battle_id, session, log[start:], 'p1', engine, own, meta, sides,
                             teacher=lambda turn: teacher_of(rec, turn))


def build(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise ValueError('choose a new output file')
    scope = DatasetScope(ledger=Path(args.ledger), manifest=Path(args.manifest))
    records = [(b, rec) for b, rec in scope.records(args.split)
               if rec.get('lineup') and rec.get('preview') and rec.get('log')]
    records = [(b, rec) for b, rec in records if (rec.get('source') or '').startswith('showdown')]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError(f'no eligible {args.split} records')
    sets = json.loads(Path(args.team).read_text())
    if args.priors == 'live':
        # No frozen train-only bundle (e.g. before any of our own battles exist): the live prior.
        from pokesim.search.lookahead import SetSampler

        engine = Lookahead(SetSampler(['M-6']))
    else:
        bundle = load_bundle(Path(args.priors))
        engine = Lookahead(sampler(bundle))
    own = {engine.sampler.base_id(to_id(s['species'])): s for s in sets}
    out.parent.mkdir(parents=True, exist_ok=True)
    total, skipped_all, started = 0, {}, time.perf_counter()
    try:
        with gzip.open(out, 'wt', encoding='utf-8') as handle:
            for n, (battle_id, rec) in enumerate(records, 1):
                rows, skipped = rows_of(battle_id, rec, sets, engine, own, args.split)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                total += len(rows)
                for k, v in skipped.items():
                    if k == 'errors':
                        skipped_all.setdefault(k, []).extend(v[:3])
                    else:
                        skipped_all[k] = skipped_all.get(k, 0) + v
                print(f'{n}/{len(records)} {battle_id} rows={len(rows)} skipped={skipped} '
                      f'{time.perf_counter() - started:.0f}s', flush=True)
    finally:
        engine.close()
    from pokesim.runtime.bundle import file_hash

    write_metadata(out, dict(scope.metadata(args.split), rows=total, battles=len(records), skipped=skipped_all,
                             rows_schema=SCHEMA,
                             prior_sha=file_hash(Path(args.priors)) if args.priors != 'live' else 'live',
                             observe_sha=file_hash(ROOT / 'services/sim-tools/observe.js'),
                             observation_sha=file_hash(ROOT / 'services/sim-tools/public_observation.js'),
                             team=str(args.team)))
    print(f'{total} rows from {len(records)} battles -> {out} (skipped {skipped_all})')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', choices=['train', 'development', 'calibration', 'test'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ledger', default=str(CURATED / 'v2/battles.jsonl'))
    ap.add_argument('--manifest', default=str(CURATED / 'split_manifest_v2.json'))
    ap.add_argument('--team', default=str(ROOT / 'data/my_teams/baton.json'))
    ap.add_argument('--priors', default=str(PRIORS))
    ap.add_argument('--limit', type=int, default=0)
    build(ap.parse_args())


if __name__ == '__main__':
    main()
