"""Local games in parallel: self-play data for learning, and paired gate matches between two search packages.

  # data generation: 4 workers, 50 games each, 3 s decisions, opponents rotating over a fixed pool
  uv run python scripts/selfplay.py play --out data/raw/selfplay/cycle1 --workers 4 --games 50 --decision-ms 3000 \
      -- --planner on --value-run runs/terminal-value-v2
  # gate: candidate vs incumbent on the same opponents and seeds, 4 workers, 60 games per arm
  uv run python scripts/selfplay.py gate --out runs/gate-net-v1 --workers 4 --games 60 \
      --candidate "--opponent-policy net:runs/opp-policy-v1/opp-seed2.json" --incumbent "" \
      -- --planner on --value-run runs/terminal-value-v2

Each worker is one `scripts/ladder.py --local` process on the local Showdown server (port 8000) with its own
seed (the opponent team sequence and the counter-picking heuristic follow the seed), writing records to
<out>/<arm>/<worker>/. The gate pairs arms by seed: both arms meet the same opponent teams in the same order.
Records have the ladder format, so pokesim.learn.records can ingest them (source 'showdown' is replaced by the
local battle tag). A worker is not started while less than MIN_FREE_MB is available.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pokesim.config import ROOT

MIN_FREE_MB = 3000
OPPONENTS = ['heuristic', 'checkpoints/ppo_baton_opp.zip', 'checkpoints/opp_baton.zip']


def free_mb() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) // 1024
    return 0


def launch(out: Path, seed: int, games: int, opponent: str, extra: list[str], decision_ms: int | None,
           search_workers: int, log: Path) -> subprocess.Popen:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'scripts/ladder.py'), '--local', '--battles', str(games), '--seed', str(seed),
           '--out', str(out)]
    if opponent != 'random':
        cmd += ['--local-opponent', opponent]
    cmd += extra
    if decision_ms:
        overrides = {'decisionMs': decision_ms}
        if '--search' in extra:
            overrides = {**json.loads(extra[extra.index('--search') + 1]), **overrides}
            cmd[cmd.index('--search', len(cmd) - len(extra)) + 1] = json.dumps(overrides)
        else:
            cmd += ['--search', json.dumps(overrides)]
    env = {'PYTHONPATH': str(ROOT / 'src'), 'POKESIM_SEARCH_WORKERS': str(search_workers),
           'OMP_NUM_THREADS': '1', 'PATH': __import__('os').environ.get('PATH', '')}
    env.update({k: v for k, v in __import__('os').environ.items() if k not in env})
    return subprocess.Popen(cmd, stdout=open(log, 'a'), stderr=subprocess.STDOUT, env=env, cwd=ROOT)


def run_arm(name: str, out: Path, workers: int, games: int, seeds: list[int], extra: list[str],
            decision_ms: int | None, search_workers: int) -> None:
    procs = []
    for k in range(workers):
        while free_mb() < MIN_FREE_MB:
            print(f'waiting for memory ({free_mb()} MB free)', flush=True)
            time.sleep(60)
        opponent = OPPONENTS[k % len(OPPONENTS)]
        procs.append(launch(out / name / f'w{k}', seeds[k], games, opponent, extra, decision_ms, search_workers,
                            out / name / f'w{k}.log'))
        time.sleep(5)
    for p in procs:
        p.wait()


def results(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.rglob('*.json')):
        if path.name == 'summary.json':
            continue
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if rec.get('result') in ('win', 'loss'):
            rows.append({'file': path.name, 'worker': path.parent.name, 'win': rec['result'] == 'win',
                         'turns': sum(ln.startswith('|turn|') for ln in rec.get('log', [])),
                         'secs': max((a.get('secs', 0) for a in rec.get('advice', [])), default=0),
                         'fallbacks': sum(bool((a.get('status') or {}).get('fallback')) for a in rec.get('advice', [])),
                         'lineup': rec.get('lineup')})
    return rows


def summarize(out: Path, arms: list[str]) -> dict:
    report = {'arms': {}}
    for arm in arms:
        rows = results(out / arm)
        wins = np.array([r['win'] for r in rows], float)
        report['arms'][arm] = {'games': len(rows), 'wins': int(wins.sum()),
                               'win_rate': float(wins.mean()) if len(rows) else None,
                               'max_decision_secs': max((r['secs'] for r in rows), default=0),
                               'fallbacks': sum(r['fallbacks'] for r in rows),
                               'by_worker': {w: [int(sum(r['win'] for r in rows if r['worker'] == w)),
                                                 sum(r['worker'] == w for r in rows)]
                                             for w in sorted({r['worker'] for r in rows})}}
    if len(arms) == 2:
        a, b = (np.array([r['win'] for r in results(out / arm)], float) for arm in arms)
        if len(a) and len(b):
            rng = np.random.default_rng(0)
            diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(4000)]
            report['difference'] = {'candidate_minus_incumbent': float(a.mean() - b.mean()),
                                    'ci90': [float(np.quantile(diffs, .05)), float(np.quantile(diffs, .95))],
                                    'ci95': [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
                                    'se_normal': float(math.sqrt(a.var() / len(a) + b.var() / len(b)))}
    (out / 'summary.json').write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('play', 'gate'):
        s = sub.add_parser(name)
        s.add_argument('--out', required=True)
        s.add_argument('--workers', type=int, default=4)
        s.add_argument('--games', type=int, default=50, help='games per worker')
        s.add_argument('--decision-ms', type=int, default=None)
        s.add_argument('--search-workers', type=int, default=2, help='POKESIM_SEARCH_WORKERS per process')
        s.add_argument('--seed', type=int, default=20261017)
        s.add_argument('extra', nargs=argparse.REMAINDER, help='after --: options for scripts/ladder.py')
    g = sub.choices['gate']
    g.add_argument('--candidate', required=True, help='extra ladder.py options of the candidate arm')
    g.add_argument('--incumbent', default='', help='extra ladder.py options of the incumbent arm')
    args = ap.parse_args()
    extra = [x for x in args.extra if x != '--']
    out = Path(args.out)
    seeds = [args.seed + k for k in range(args.workers)]
    if args.command == 'play':
        run_arm('play', out, args.workers, args.games, seeds, extra, args.decision_ms, args.search_workers)
        print(json.dumps(summarize(out, ['play']), indent=1))
        return
    for arm, options in (('candidate', args.candidate), ('incumbent', args.incumbent)):
        run_arm(arm, out, args.workers, args.games, seeds, extra + options.split(), args.decision_ms,
                args.search_workers)
        print(json.dumps(summarize(out, [arm])['arms'], indent=1), flush=True)
    print(json.dumps(summarize(out, ['candidate', 'incumbent']), indent=1))


if __name__ == '__main__':
    main()
