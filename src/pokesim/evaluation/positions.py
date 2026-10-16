"""A fixed set of real ladder positions, an offline oracle, and the regret of a search setting against it.

  uv run python -m pokesim.evaluation.positions build  --out data/processed/positions/v1 [--since 20261016-1600]
  uv run python -m pokesim.evaluation.positions oracle --set data/processed/positions/v1 --streams 8 --budget-ms 60000
  uv run python -m pokesim.evaluation.positions eval   --set data/processed/positions/v1 --name base [--search '{...}']
  uv run python -m pokesim.evaluation.positions report --set data/processed/positions/v1

build captures, for a stratified sample of recorded decisions (the biggest evaluation drops in losses, other loss
turns, win turns), exactly what the live search would receive: the sampled worlds, the observed history and the
worker settings (Lookahead.query(capture=True)). Nothing from later in the game is used.
oracle searches every position much longer than live play, over several independent random streams, and keeps
each root action's objective (the planner's (1 - rho) * expected + rho * worst). eval replays the positions at the
live budget with a candidate setting; its regret is the oracle objective of the oracle's best action minus that of
the action the candidate plays. The oracle shares the rollout policies' biases, so regret measures search quality
under those policies, not the truth.
"""
import argparse
import copy
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pokesim.config import ROOT

KINDS = ('blunder', 'loss', 'win')


def objective_of(entry: dict, rho: float) -> float | None:
    """The planner's objective of the action played at a recorded decision (None without root evidence)."""
    evidence = (entry.get('status') or {}).get('search', {}).get('root_evidence') or []
    for row in evidence:
        if row['label'] == entry.get('chosen'):
            return (1 - rho) * row['expected'] + rho * row['worst']
    return None


def decisions(rec: dict) -> list[dict]:
    """Normal (non forced) decisions of a record with a terminal search and a played action."""
    rho = ((rec.get('versions') or {}).get('planner') or {}).get('rho', .3)
    out = []
    for entry in rec.get('advice') or []:
        if entry.get('force') or not entry.get('chosen') or entry['chosen'] == 'random':
            continue
        status = (entry.get('status') or {}).get('search') or {}
        if not status.get('terminal_search'):
            continue
        value = objective_of(entry, rho)
        if value is None:
            continue
        out.append({'turn': entry['turn'], 'chosen': entry['chosen'], 'objective': value, 'rho': rho})
    return out


def candidates(records: list[tuple[str, dict]]) -> dict[str, list[dict]]:
    """Positions by kind. blunder: per loss the decision before the largest drop of the played objective."""
    out: dict[str, list[dict]] = {k: [] for k in KINDS}
    for stem, rec in records:
        rows = decisions(rec)
        if not rows:
            continue
        seen = set()
        for i, row in enumerate(rows):
            if row['turn'] in seen:
                continue
            seen.add(row['turn'])
            row = dict(row, record=stem, result=rec['result'])
            row['drop'] = row['objective'] - rows[i + 1]['objective'] if i + 1 < len(rows) else 0.
            out['loss' if rec['result'] == 'loss' else 'win'].append(row)
    for stem in {r['record'] for r in out['loss']}:
        rows = [r for r in out['loss'] if r['record'] == stem]
        worst = max(rows, key=lambda r: r['drop'])
        out['loss'].remove(worst)
        out['blunder'].append(dict(worst, kind='blunder'))
    out['blunder'].sort(key=lambda r: -r['drop'])
    for kind in ('loss', 'win'):
        for r in out[kind]:
            r['kind'] = kind
    return out


def stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n rows spread over early / middle / late turns (equal shares, remainders to the fullest bins)."""
    if n >= len(rows):
        return list(rows)
    bins = {'early': [r for r in rows if r['turn'] <= 3], 'middle': [r for r in rows if 3 < r['turn'] <= 8],
            'late': [r for r in rows if r['turn'] > 8]}
    chosen: list[dict] = []
    share = n // 3
    for key in bins:
        rng.shuffle(bins[key])
        chosen.extend(bins[key][:share])
        bins[key] = bins[key][share:]
    rest = [r for key in bins for r in bins[key]]
    rng.shuffle(rest)
    return chosen + rest[:n - len(chosen)]


def load_records(directory: Path, since: str, until: str, commit: str | None) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(directory.glob('*.json')):
        rec = json.loads(path.read_text())
        if not since <= rec.get('started', '') <= until or rec.get('result') not in ('win', 'loss'):
            continue
        if commit and not ((rec.get('versions') or {}).get('code') or {}).get('commit', '').startswith(commit):
            continue
        if rec.get('preview') and rec.get('lineup') and rec.get('log'):
            out.append((path.stem, rec))
    return out


def build(args) -> None:
    from pokesim.evaluation.shadow import position
    from pokesim.live.advisor import Advisor
    from pokesim.live.opponent import DEFAULT_POLICY, OpponentPolicy
    from pokesim.runtime.bundle import describe

    out = Path(args.out)
    if out.exists():
        raise ValueError('choose a new position set directory')
    records = load_records(Path(args.records), args.since, args.until, args.commit)
    if not records:
        raise ValueError('no eligible records')
    rng = random.Random(args.seed)
    pool = candidates(records)
    picked = (pool['blunder'][:args.blunders] + stratified(pool['loss'], args.losses, rng) +
              stratified(pool['win'], args.wins, rng))
    sets = json.loads(Path(args.team).read_text())
    advisor = Advisor(sets, policy=None, planner={'mode': 'on'}, opponent_policy=None, belief=True,
                      value_run=args.value_run)
    opponent = OpponentPolicy(DEFAULT_POLICY) if DEFAULT_POLICY.exists() else None
    by_stem = dict(records)
    (out / 'positions').mkdir(parents=True)
    saved, skipped = [], []
    try:
        for row in picked:
            rec = by_stem[row['record']]
            try:
                session, battle = position(rec, sets, row['turn'])
                belief = advisor.beliefs.get(row['record'], session.log, 'p1')
                request = advisor.engine.query(battle, advisor.own_sets, None, advisor.planner['search'],
                                               belief=belief, log=session.log, our_role='p1', value=True,
                                               capture=True)
            except Exception as e:  # noqa: BLE001 - an unreproducible position is reported, not silently dropped
                skipped.append({**row, 'error': repr(e)[:200]})
                continue
            request.pop('value_model', None)
            cloned = dict(opponent.predict(session.log, 'p1')) if opponent else {}
            pid = f'{len(saved) + 1:03d}'
            fixture = dict(request, source=f"{row['record']} T{row['turn']}; observed prefix only; "
                                           'opponent set particles are hypotheses',
                           meta={**row, 'id': pid, 'last_move': session.last_move('p2'), 'cloned': cloned,
                                 'legal': [lab for lab, _ in (rec['advice'] and next(
                                     a for a in rec['advice'] if a['turn'] == row['turn'] and not a.get('force'))
                                     .get('actions') or [])]})
            (out / 'positions' / f'{pid}.json').write_text(json.dumps(fixture))
            saved.append({k: row[k] for k in ('record', 'turn', 'kind', 'result', 'chosen', 'objective', 'drop')}
                         | {'id': pid})
            print(pid, row['kind'], row['record'], f"T{row['turn']}", row['chosen'],
                  f"obj={row['objective']:+.2f} drop={row['drop']:+.2f}", flush=True)
    finally:
        advisor.close()
    manifest = {'schema_version': 1, 'records_dir': str(args.records), 'since': args.since, 'until': args.until,
                'commit': args.commit, 'seed': args.seed, 'team': str(args.team), 'value_run': str(args.value_run),
                'counts': {k: sum(r['kind'] == k for r in saved) for k in KINDS},
                'available': {k: len(v) for k, v in pool.items()}, 'positions': saved, 'skipped': skipped,
                'search': advisor.planner['search'] | {'value_model': None},
                'versions': describe(None, None, planner=advisor.planner, belief=True),
                'note': 'positions are observed prefixes of real battles; worlds are sampled hypotheses. '
                        'These records are development data from now on, not an unseen test.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"{len(saved)} positions ({manifest['counts']}), {len(skipped)} skipped -> {out}")


def merge_results(runs: list[list[dict]]) -> list[dict]:
    """Per world, the visit-weighted mean of several independent searches' payoffs (prior-only cells averaged)."""
    merged = copy.deepcopy(runs[0])
    for w, world in enumerate(merged):
        p1, p2 = world['p1'], world['p2']
        for i in range(len(p1)):
            for j in range(len(p2)):
                total = visits = 0.
                speed = 0.
                priors = []
                for run in runs:
                    cell = run[w]
                    n = cell['visits'][i][j]
                    v = cell['payoff'][i][j]
                    if v is None:
                        continue
                    if n:
                        total += n * v
                        speed += n * cell['winSpeed'][i][j]
                        visits += n
                    else:
                        priors.append((v, cell['winSpeed'][i][j]))
                if visits:
                    world['payoff'][i][j], world['winSpeed'][i][j] = total / visits, speed / visits
                elif priors:
                    world['payoff'][i][j] = float(np.mean([p for p, _ in priors]))
                    world['winSpeed'][i][j] = float(np.mean([s for _, s in priors]))
                world['visits'][i][j] = int(visits)
                world['invalid'][i][j] = sum(run[w]['invalid'][i][j] for run in runs)
        summary = dict(world['terminal_search'])
        for key in ('simulations', 'terminal', 'learned', 'censored', 'failed', 'unfinished_at_deadline'):
            summary[key] = sum(run[w]['terminal_search'].get(key, 0) for run in runs)
        summary['streams'] = len(runs)
        world['terminal_search'] = summary
    return merged


def objectives(results: list[dict], fixture: dict) -> dict:
    """Each root action's objective, expected and worst value, and the action the planner would play."""
    from pokesim.live.advisor import OPPONENT_MODEL
    from pokesim.search.planner import plan

    meta = fixture['meta']
    advice = plan(results, OPPONENT_MODEL, meta.get('last_move'), meta.get('rho', .3), meta.get('cloned') or None,
                  .5, pure=True)
    rho = meta.get('rho', .3)
    rows = {r['label']: r for r in advice.status['root_evidence']}
    return {'objective': {lab: (1 - rho) * r['expected'] + rho * r['worst'] for lab, r in rows.items()},
            'expected': {lab: r['expected'] for lab, r in rows.items()},
            'worst': {lab: r['worst'] for lab, r in rows.items()},
            'trials': {lab: r['trials'] for lab, r in rows.items()},
            'chosen': advice.ranked()[0][0], 'solver': advice.status['solver']}


def predictor_of(fixture: dict):
    """The opponent prediction the live planner used at this position (its last move and cloned policy)."""
    from pokesim.live.advisor import OPPONENT_MODEL
    from pokesim.search.planner import predict

    meta = fixture['meta']

    def predictor(payoff, labels, prior=None):
        return predict(payoff, labels, OPPONENT_MODEL, meta.get('last_move'), meta.get('cloned') or None, .5, prior)
    return predictor


def run_search(engine, fixture: dict, budget_ms: int, stream: int, overrides: dict | None, max_budget: float,
               predicted_columns: bool = True):
    from pokesim.search.terminal import search

    request = {k: v for k, v in fixture.items() if k not in ('source', 'meta', 'expected')}
    request.update(overrides or {})
    request['decisionMs'] = budget_ms
    if stream:
        request['stream'] = stream
    return search(engine, request, max_budget=max_budget,
                  predictor=predictor_of(fixture) if predicted_columns else None)


def engines(n: int, value_run: Path):
    from pokesim.search.lookahead import Lookahead
    from pokesim.value.runtime import ValueEnsemble

    ensemble = ValueEnsemble(value_run)
    return [Lookahead(None, value_model=ensemble) for _ in range(n)]


def position_files(root: Path) -> list[Path]:
    return sorted((root / 'positions').glob('*.json'))


def oracle(args) -> None:
    root = Path(args.set)
    out = root / 'oracle'
    out.mkdir(exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    pool = engines(args.streams, Path(args.value_run or manifest['value_run']))
    overrides = json.loads(args.search) if args.search else {}
    max_budget = args.budget_ms / 1000 + 1
    try:
        for path in position_files(root):
            target = out / path.name
            if target.exists():
                continue
            fixture = json.loads(path.read_text())
            start = time.perf_counter()
            with ThreadPoolExecutor(args.streams) as ex:
                runs = list(ex.map(lambda k: run_search(pool[k], fixture, args.budget_ms, k + 1, overrides,
                                                        max_budget, predicted_columns=not args.uniform_columns),
                                   range(args.streams)))
            merged = merge_results(runs)
            result = objectives(merged, fixture) | {
                'budget_ms': args.budget_ms, 'streams': args.streams, 'overrides': overrides,
                'simulations': merged[0]['terminal_search']['simulations'],
                'seconds': time.perf_counter() - start,
                'per_stream': [objectives(run, fixture)['objective'] for run in runs]}
            target.write_text(json.dumps(result, indent=1))
            best = max(result['objective'], key=result['objective'].get)
            print(path.stem, fixture['meta']['kind'], f"best={best} played={fixture['meta']['chosen']}",
                  f"sims={result['simulations']} {result['seconds']:.0f}s", flush=True)
    finally:
        for engine in pool:
            engine.close()


def evaluate(args) -> None:
    root = Path(args.set)
    manifest = json.loads((root / 'manifest.json').read_text())
    out = root / 'eval' / args.name
    if out.exists():
        raise ValueError('choose a new evaluation name')
    out.mkdir(parents=True)
    overrides = json.loads(args.search) if args.search else {}
    pool = engines(args.jobs, Path(args.value_run or manifest['value_run']))
    files = [p for p in position_files(root) if (root / 'oracle' / p.name).exists()]
    if not files:
        raise ValueError('run the oracle first')

    def one(index: int) -> dict:
        path = files[index]
        fixture = json.loads(path.read_text())
        start = time.perf_counter()
        results = run_search(pool[index % args.jobs], fixture, args.budget_ms, args.stream, overrides, 9.,
                             predicted_columns=not args.uniform_columns)
        row = objectives(results, fixture)
        row.update(id=path.stem, kind=fixture['meta']['kind'], played=fixture['meta']['chosen'],
                   seconds=time.perf_counter() - start,
                   simulations=results[0]['terminal_search']['simulations'],
                   tree_decisions=results[0]['terminal_search']['max_tree_decisions'])
        return row

    rows = []
    try:
        # One engine per job: positions are sharded so each engine serves one position at a time.
        with ThreadPoolExecutor(args.jobs) as ex:
            for shard in range(0, len(files), args.jobs):
                rows.extend(ex.map(one, range(shard, min(len(files), shard + args.jobs))))
                print(f'{len(rows)}/{len(files)}', flush=True)
    finally:
        for engine in pool:
            engine.close()
    for row in rows:
        oracle_row = json.loads((root / 'oracle' / f"{row['id']}.json").read_text())
        row['regret'] = regret(oracle_row['objective'], row['chosen'])
        row['agree'] = row['chosen'] == max(oracle_row['objective'], key=oracle_row['objective'].get)
        row['played_regret'] = regret(oracle_row['objective'], row['played'])
        (out / f"{row['id']}.json").write_text(json.dumps(row, indent=1))
    report = summarize(rows) | {'name': args.name, 'overrides': overrides, 'budget_ms': args.budget_ms,
                                'stream': args.stream, 'set': str(root)}
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'by_kind'}, indent=1))
    print(json.dumps(report['by_kind'], indent=1))


def regret(oracle_objective: dict[str, float], chosen: str) -> float | None:
    if chosen not in oracle_objective:
        return None
    return max(oracle_objective.values()) - oracle_objective[chosen]


def bootstrap(values: list[float], samples: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), (samples, len(v)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get('regret') is not None]
    by_kind = {}
    for kind in KINDS:
        part = [r for r in scored if r['kind'] == kind]
        if part:
            by_kind[kind] = {'positions': len(part), 'regret': float(np.mean([r['regret'] for r in part])),
                             'agree': float(np.mean([r['agree'] for r in part])),
                             'played_regret': float(np.mean([r['played_regret'] for r in part
                                                             if r['played_regret'] is not None] or [np.nan]))}
    return {'positions': len(rows), 'scored': len(scored),
            'regret': float(np.mean([r['regret'] for r in scored])) if scored else None,
            'regret_ci95': bootstrap([r['regret'] for r in scored]) if len(scored) > 1 else None,
            'agree': float(np.mean([r['agree'] for r in scored])) if scored else None,
            'simulations': float(np.median([r['simulations'] for r in rows])) if rows else None,
            'tree_decisions': float(np.median([r['tree_decisions'] for r in rows])) if rows else None,
            'seconds': float(np.max([r['seconds'] for r in rows])) if rows else None,
            'by_kind': by_kind}


def report(args) -> None:
    root = Path(args.set)
    names = sorted(p.parent.name for p in (root / 'eval').glob('*/report.json'))
    print(f"{'name':24s} {'regret':>8s} {'ci95':>18s} {'agree':>6s} {'sims':>6s} {'depth':>5s}  by kind")
    for name in names:
        r = json.loads((root / 'eval' / name / 'report.json').read_text())
        ci = r['regret_ci95'] or [float('nan')] * 2
        kinds = ' '.join(f"{k}={v['regret']:.3f}/{v['agree']:.0%}" for k, v in r['by_kind'].items())
        print(f"{name:24s} {r['regret']:8.4f} [{ci[0]:7.4f},{ci[1]:7.4f}] {r['agree']:6.0%} "
              f"{r['simulations']:6.0f} {r['tree_decisions']:5.1f}  {kinds}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--records', default=str(ROOT / 'data/raw/ladder'))
    b.add_argument('--since', default='')
    b.add_argument('--until', default='99999999-999999')
    b.add_argument('--commit', default=None, help='only records played by this code commit (prefix)')
    b.add_argument('--team', default=str(ROOT / 'data/my_teams/baton.json'))
    b.add_argument('--value-run', default=str(ROOT / 'runs/terminal-value-v2'))
    b.add_argument('--blunders', type=int, default=57)
    b.add_argument('--losses', type=int, default=100)
    b.add_argument('--wins', type=int, default=143)
    b.add_argument('--seed', type=int, default=20261016)
    o = sub.add_parser('oracle')
    o.add_argument('--set', required=True)
    o.add_argument('--streams', type=int, default=8)
    o.add_argument('--budget-ms', type=int, default=60000)
    o.add_argument('--search', default=None, help='JSON worker settings for the oracle (e.g. prior visits)')
    o.add_argument('--value-run', default=None)
    o.add_argument('--uniform-columns', action='store_true',
                   help='cover the opponent replies evenly instead of by the live prediction')
    e = sub.add_parser('eval')
    e.add_argument('--set', required=True)
    e.add_argument('--name', required=True)
    e.add_argument('--search', default=None, help='JSON worker settings overriding the captured ones')
    e.add_argument('--budget-ms', type=int, default=8500)
    e.add_argument('--stream', type=int, default=0, help='0: the live random stream')
    e.add_argument('--jobs', type=int, default=4)
    e.add_argument('--value-run', default=None)
    e.add_argument('--uniform-columns', action='store_true',
                   help='cover the opponent replies evenly instead of by the live prediction')
    r = sub.add_parser('report')
    r.add_argument('--set', required=True)
    args = ap.parse_args()
    {'build': build, 'oracle': oracle, 'eval': evaluate, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
