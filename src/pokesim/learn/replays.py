"""Actor views and chosen actions from public Showdown replays, from both players' sides.

  uv run python -m pokesim.learn.replays --out data/processed/learn/replays-v1.jsonl.gz [--shard 0/4] [--limit N]

Each replay is replayed twice, once as each player. The actor's own team is what the whole battle revealed of it,
filled in from the training prior (a player knows its own set, so this is the closest stand-in); the other side is
only what had been seen by the start of the turn, plus a hypothesis for the rest, as in pokesim.learn.records.
Our own ladder battles are left out (they are in the records datasets). Rows have the same fields as the ladder
rows, with source 'replay', the actor's rating and the replay id; no split (assign by replay id when training).
"""
import argparse
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

from pokesim.config import ROOT
from pokesim.data.dataset import write_metadata
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.learn.records import PRIORS, SCHEMA, ratings, rows_from_session
from pokesim.rl.myteam import to_id
from pokesim.search.lookahead import Lookahead

REPLAYS = ROOT / 'data/raw/replays'


def own_games() -> set[str]:
    """Replay ids of battles our own ladder player took part in (left out as training rows elsewhere)."""
    ids = set()
    for path in (ROOT / 'data/raw/ladder').glob('*.json'):
        try:
            source = json.loads(path.read_text()).get('source') or ''
        except (OSError, ValueError):
            continue
        if m := re.search(r'battle-([a-z0-9]+-\d+)', source):
            ids.add(m.group(1))
    return ids


def replay_rows(replay_id: str, log: str, sides, engine: Lookahead) -> tuple[list[dict], dict]:
    from pokesim.rl.human import open_session

    lines = [ln for ln in log.split('\n') if ln.startswith('|')]
    winner = next((ln.split('|')[2] for ln in lines if ln.startswith('|win|')), None)
    names = {ln.split('|')[2]: ln.split('|')[3] for ln in lines
             if ln.startswith('|player|') and len(ln.split('|')) > 3 and ln.split('|')[3]}
    rate = ratings(lines)
    rows, skipped = [], {}
    for role in ('p1', 'p2'):
        try:
            sets = sides.sets(log, role)
            session, body = open_session(lines, role, sets, sides.species)
        except Exception as e:  # noqa: BLE001 - a side whose sets or lineup cannot be rebuilt
            skipped[f'{role}_unopened'] = repr(e)[:120]
            continue
        own = {engine.sampler.base_id(to_id(s['species'])): s for s in sets}
        outcome = .5 if winner is None else float(winner == names.get(role, role))
        meta = {'source': 'replay', 'record': replay_id, 'started': replay_id, 'split': None, 'outcome': outcome,
                'rating': rate, 'opponent': names.get('p2' if role == 'p1' else 'p1'), 'actor': role}
        part, part_skipped = rows_from_session(f'replay:{replay_id}:{role}', session, body, role, engine, own,
                                               meta, sides=('p1',))
        rows.extend(part)
        for k, v in part_skipped.items():
            if k != 'errors':
                skipped[k] = skipped.get(k, 0) + v
    return rows, skipped


def build(args) -> None:
    from pokesim.rl.human import Sides

    out = Path(args.out)
    if out.exists():
        raise ValueError('choose a new output file')
    shard, shards = (int(x) for x in args.shard.split('/'))
    paths = sorted(REPLAYS.glob('*.json'))
    excluded = own_games()
    paths = [p for p in paths if not any(p.stem.endswith(x) for x in excluded)]
    paths = [p for i, p in enumerate(paths) if i % shards == shard]
    if args.limit:
        paths = paths[:args.limit]
    bundle = load_bundle(Path(args.priors))
    engine = Lookahead(sampler(bundle))
    sides = Sides(sampler=engine.sampler)
    out.parent.mkdir(parents=True, exist_ok=True)
    total, skipped_all, started = 0, {}, time.perf_counter()
    try:
        with gzip.open(out, 'wt', encoding='utf-8') as handle:
            for n, path in enumerate(paths, 1):
                data = json.loads(path.read_text())
                log = data['log'] if isinstance(data, dict) else data
                replay_id = path.stem
                rows, skipped = replay_rows(replay_id, log, sides, engine)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                total += len(rows)
                for k, v in skipped.items():
                    skipped_all[k] = skipped_all.get(k, 0) + (1 if isinstance(v, str) else v)
                if n % 25 == 0 or n == len(paths):
                    print(f'{n}/{len(paths)} rows={total} skipped={skipped_all} {time.perf_counter() - started:.0f}s',
                          flush=True)
    finally:
        engine.close()
    from pokesim.runtime.bundle import file_hash

    write_metadata(out, {'schema_version': 2, 'rows_schema': SCHEMA, 'rows': total, 'replays': len(paths),
                         'shard': args.shard, 'excluded_own_games': len(excluded), 'skipped': skipped_all,
                         'prior_sha': file_hash(Path(args.priors)),
                         'observe_sha': file_hash(ROOT / 'services/sim-tools/observe.js'),
                         'replays_sha': hashlib.sha256('\n'.join(p.name for p in paths).encode()).hexdigest()[:16]})
    print(f'{total} rows from {len(paths)} replays -> {out} (skipped {skipped_all})')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--priors', default=str(PRIORS.with_name('priors-v2.json')))
    ap.add_argument('--shard', default='0/1', help='k/n: this process takes every n-th replay starting at k')
    ap.add_argument('--limit', type=int, default=0)
    build(ap.parse_args())


if __name__ == '__main__':
    main()
