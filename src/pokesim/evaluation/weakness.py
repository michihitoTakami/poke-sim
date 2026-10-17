"""Audit saved ladder games, with battle-level denominators and turn-level evidence.

python -m pokesim.evaluation.weakness --out docs/audits/20261017-weakness.json

Appearances and moves correlate with the final result; they are not counterfactual proof of a bad decision.
Mega/battle formes count once per base species. Unknown results remain outside win/loss statistics.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pokesim.config import ROOT

SETUP = {'Calm Mind', 'Nasty Plot', 'Dragon Dance', 'Iron Defense', 'Cosmic Power', 'Tidy Up', 'Swords Dance'}
PASSIVE = SETUP | {'Substitute', 'Protect', 'Recover', 'Synthesis', 'Slack Off', 'Roost', 'Moonlight', 'Baton Pass'}


def ident(name):
    return ''.join(c for c in name.lower() if c.isalnum())


def species_aliases(dex):
    out = {}
    for mon in dex['species']:
        base = mon.get('battleOnly') or mon.get('baseSpecies') or mon['name']
        # Regional formes keep their identity; only battle transformations are merged.
        if not mon.get('battleOnly') and '-Mega' not in mon['name']:
            base = mon['name']
        if isinstance(base, list):
            base = base[0]
        out[ident(mon['name'])] = base
    return out


def analyse_record(rec, aliases):
    def base(name):
        return aliases.get(ident(name), name)

    selected = {base(x) for x in rec.get('lineup') or []}
    names, seen, preview, moves = {}, set(), set(), set()
    active, fainted, own_seen = {}, set(), set()
    rows, turn, last_move = [], 0, None
    turn_moves = defaultdict(dict)
    failed, kos, markers = [], [], []
    weather = None
    for line in rec.get('log') or []:
        p = line.split('|')
        if len(p) < 3:
            continue
        kind = p[1]
        side = p[2][:2]
        if kind == 'turn':
            turn, last_move = int(p[2]), None
            rows.append(turn)
        elif kind == '-weather':
            weather = p[2]
        elif kind == 'poke' and side == 'p2' and len(p) > 3:
            preview.add(base(p[3].split(',')[0]))
        elif kind in ('switch', 'drag') and len(p) > 3:
            species = base(p[3].split(',')[0])
            names[p[2]] = species
            active[side] = species
            if side == 'p2':
                seen.add(species)
            elif side == 'p1':
                own_seen.add(species)
            last_move = None
        elif kind == 'move' and len(p) > 3 and '[from]' not in line:
            last_move = {'side': side, 'species': active.get(side), 'move': p[3], 'turn': turn}
            turn_moves[turn].setdefault(side, dict(last_move))
            if side == 'p2':
                moves.add(p[3])
                if p[3] == 'Electro Shot' and weather == 'RainDance':
                    markers.append({'kind': 'rain_electro_shot', **last_move})
        elif kind == '-fail' and side == 'p1' and last_move and last_move['side'] == 'p1':
            failed.append(dict(last_move))
        elif kind == '-damage' and side == 'p1' and len(p) > 3 and p[3].startswith('0'):
            source = 'residual_or_unknown'
            move, attacker = None, None
            if '[from]' not in line and last_move and last_move['side'] == 'p2':
                source, move, attacker = 'direct_move', last_move['move'], last_move['species']
            kos.append({'turn': turn, 'victim': names.get(p[2], active.get('p1')),
                        'source': source, 'move': move, 'attacker': attacker, 'protocol': line})
        elif kind == 'faint' and side == 'p1':
            fainted.add(names.get(p[2], active.get('p1')))
            remaining = selected - fainted
            if remaining == {'Scolipede'}:
                markers.append({'kind': 'last_scolipede', 'turn': turn})
        elif kind == '-start' and side == 'p1' and len(p) > 3 and p[3] in ('Encore', 'Curse'):
            markers.append({'kind': ident(p[3]), 'turn': turn, 'victim': active.get('p1')})
    # A drag was already processed as a switch above; collect it independently.
    turn = 0
    for line in rec.get('log') or []:
        p = line.split('|')
        if p[1:2] == ['turn']:
            turn = int(p[2])
        elif p[1:2] == ['drag'] and p[2].startswith('p1'):
            markers.append({'kind': 'phazed', 'turn': turn})
    for t, actions in turn_moves.items():
        ours, theirs = actions.get('p1'), actions.get('p2')
        if ours and theirs and theirs['move'] in SETUP and ours['move'] in PASSIVE:
            markers.append({'kind': 'opponent_setup_during_passive_move', 'turn': t,
                            'our_move': ours['move'], 'opponent_move': theirs['move']})
    mismatch = bool(own_seen - selected)
    if mismatch:
        markers = [m for m in markers if m['kind'] != 'last_scolipede']
    return {'battle_id': rec.get('source'), 'started': rec.get('started'), 'result': rec.get('result'),
            'opponent': rec.get('opponent'), 'code': (rec.get('versions') or {}).get('code'),
            'turns': len(rows), 'lineup': rec.get('lineup'), 'observed_own': sorted(own_seen),
            'lineup_mismatch': mismatch, 'opponent_preview': sorted(preview), 'opponent_seen': sorted(seen),
            'opponent_moves': sorted(moves), 'own_faints': kos, 'failed_moves': failed, 'markers': markers,
            'random_fallbacks': sum(a.get('chosen') == 'random' for a in rec.get('advice') or [])}


def summarise(battles):
    known = [b for b in battles if b['result'] in ('win', 'loss')]

    def table(field):
        counts = defaultdict(Counter)
        for battle in known:
            for item in set(battle[field]):
                counts[item][battle['result']] += 1
        return [{'name': item, 'battles': c.total(), 'wins': c['win'], 'losses': c['loss'],
                 'win_rate': c['win'] / c.total()}
                for item, c in sorted(counts.items(), key=lambda kv: (-kv[1]['loss'], kv[1]['win'], kv[0]))]

    markers = defaultdict(Counter)
    for battle in known:
        for kind in {m['kind'] for m in battle['markers']}:
            markers[kind][battle['result']] += 1
    ko_moves = Counter(k['move'] for b in known for k in b['own_faints'] if k['source'] == 'direct_move')
    return {'battles': len(battles), 'known_results': len(known),
            'results': dict(Counter(b['result'] for b in battles)),
            'by_preview': table('opponent_preview'), 'by_seen': table('opponent_seen'),
            'by_move': table('opponent_moves'), 'direct_ko_moves': dict(ko_moves.most_common()),
            'marker_battles': {k: dict(v) for k, v in sorted(markers.items())},
            'lineup_mismatches': [b['battle_id'] for b in battles if b['lineup_mismatch']],
            'random_fallbacks': sum(b['random_fallbacks'] for b in battles)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--since', default='')
    ap.add_argument('--until', default='', help='inclusive snapshot cutoff (YYYYMMDD-HHMMSS)')
    ap.add_argument('--dex', type=Path, default=ROOT / 'data/processed/public_value/training-dex-v1.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    aliases = species_aliases(json.loads(args.dex.read_text()))
    battles = []
    for path in sorted(args.dir.glob('*.json')):
        rec = json.loads(path.read_text())
        if rec.get('started', '') < args.since or not rec.get('source', '').startswith('showdown '):
            continue
        if args.until and rec.get('started', '') > args.until:
            continue
        battle = analyse_record(rec, aliases)
        battle['record'] = str(path)
        battle['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        battles.append(battle)
    report = {'schema_version': 1, 'scope': 'saved Showdown ladder records',
              'interpretation': 'Conditional battle outcomes and observed events, not causal move ratings. '
                                'Direct KOs exclude residual damage. Small samples and repeated opponents remain.',
              'summary': summarise(battles), 'battles': battles}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
