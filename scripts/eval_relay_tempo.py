"""Fixed-prefix ladder replay and paired terminal diagnostics; never sends live orders."""
import argparse
import copy
import io
import json
import subprocess
import tarfile
from pathlib import Path

from pokesim.belief.sets import Beliefs
from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.evaluation.branch_matches import atomic_json, file_sha, hash_seed, summarize, tree_sha
from pokesim.evaluation.shadow import position
from pokesim.search.evaluation import leaf_value
from pokesim.search.lookahead import Lookahead, Worker, to_id
from pokesim.search.planner import plan
from pokesim.search.terminal import search
from pokesim.value.runtime import ValueEnsemble


class ReplayWorker(Worker):
    def __init__(self, script, root=None):
        self.script, self.root = script, root
        super().__init__()

    def start(self):
        args = ['node', str(self.script), str(SHOWDOWN_DIR)]
        if self.root is not None:
            args.append(str(self.root))
        self.proc = subprocess.Popen(args,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def fixed_search(root, request, value, trials):
    worker = ReplayWorker(ROOT / 'scripts/replay_fixed_search.js', root)
    try:
        reply = worker.query({'op': 'start', 'request': request}, 120.)
        remaining = trials
        while True:
            probabilities = value.probabilities([p['observation'] for p in reply['pending']])
            feedback = [{'id': row['id'], 'p': (leaf_value(row['observation'], float(p)) + 1) / 2}
                        for row, p in zip(reply['pending'], probabilities, strict=True)]
            count = min(8, remaining)
            reply = worker.query({'op': 'advance', 'feedback': feedback, 'trials': count}, 120.)
            remaining -= count
            if count == 0:
                return reply['results']
    finally:
        worker.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--priors', type=Path, required=True)
    ap.add_argument('--value-run', type=Path, default=ROOT / 'runs/value-replays-v1')
    ap.add_argument('--opponent-policy', type=Path, default=ROOT / 'runs/opp-replays-v1/opp-seed0.json')
    ap.add_argument('--catalog', type=Path)
    ap.add_argument('--baseline-ref', default='365306c')
    ap.add_argument('--worlds', type=int, default=4)
    ap.add_argument('--budget-ms', type=int, default=4500)
    ap.add_argument('--trials', type=int, default=512, help='fixed completed trials; 0 uses wall-clock search')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.trials < 0 or args.worlds < 1:
        ap.error('trials must be nonnegative and worlds must be positive')
    hash_seed()
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = args.out.resolve() / 'baseline'
    baseline.mkdir()
    archive = subprocess.check_output(['git', 'archive', args.baseline_ref, 'services/sim-tools'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(baseline, filter='data')
    value = ValueEnsemble(args.value_run)
    engines = {name: Lookahead(sampler(load_bundle(args.priors)), samples=args.worlds, value_model=value)
               for name in ('before', 'after')}
    if args.catalog:
        from pokesim.belief.team_catalog import TeamCatalog

        for engine in engines.values():
            engine.sampler.catalog = TeamCatalog.load(engine.sampler.base_id, path=args.catalog)
    engines['before'].worker.kill()
    engines['before'].worker = ReplayWorker(baseline / 'services/sim-tools/lookahead.js')
    beliefs = Beliefs(json.loads((ROOT / 'data/my_teams/baton.json').read_text()), engines['after'].sampler)
    sets = beliefs.sets
    own = {engines['after'].sampler.base_id(to_id(s['species'])): s for s in sets}
    report = {'scope': 'development diagnostics; observed prefixes and hypothetical sets; mixed control sources',
              'baseline_ref': args.baseline_ref, 'settings': {k: str(v) for k, v in vars(args).items()},
              'priors_sha': file_sha(args.priors),
              'catalog_sha': file_sha(args.catalog) if args.catalog else None, 'value': value.version,
              'worker_sha': {name: tree_sha(directory, {'.js'}) for name, directory in
                             [('before', baseline / 'services/sim-tools'), ('after', ROOT / 'services/sim-tools')]},
              'cases': []}
    try:
        for case in json.loads(args.cases.read_text()):
            path = next(args.records.glob(case['stamp'] + '*.json'))
            rec = json.loads(path.read_text())
            session, battle = position(rec, sets, case['turn'])
            belief = beliefs.get(path.name, session.log, 'p1')
            settings = copy.deepcopy(rec['versions']['planner']['search'])
            settings.pop('value_model', None)
            settings.update(workers=1, decisionMs=args.budget_ms)
            settings['policyNets'] = {'opp': str(args.opponent_policy.resolve())}
            request = engines['after'].query(battle, own, search=settings, belief=belief,
                log=session.log, value=True, capture=True)
            atomic_json(args.out / f'{case["stamp"]}-T{case["turn"]}.request.json', request)
            row = {'record': path.name, 'sha256': file_sha(path), 'turn': case['turn'], 'variants': {}}
            for name, engine in engines.items():
                matrices = fixed_search(baseline if name == 'before' else ROOT, request, value, args.trials) \
                    if args.trials else search(engine, copy.deepcopy(request))
                result = plan(matrices, pure=True)
                row['variants'][name] = {'chosen': result.ranked()[0][0],
                    'evidence': result.status.get('root_evidence'), 'search': result.status['terminal_search']}
                print(path.name, case['turn'], name, row['variants'][name]['chosen'], flush=True)
            actions = sorted(set(v['chosen'] for v in row['variants'].values()) | set(case.get('actions', [])))
            trial = {k: request[k] for k in ('format', 'samples', 'observationLog')}
            trial['branches'] = {'informationScope': 'public_observation', 'observationLog': session.log,
                'seeds': [0, 1], 'maxTurns': 160, 'actions': actions,
                'continuations': ['planned/tactical', 'planned/disrupt', 'planned/expert', 'planned/planned']}
            row['terminal'] = {}
            for name, engine in engines.items():
                engine.query_timeout = 120.
                row['terminal'][name] = summarize(engine._query(trial))
            report['cases'].append(row)
            atomic_json(args.out / 'report.json', report)
    finally:
        beliefs.close()
        for engine in engines.values():
            engine.close()


if __name__ == '__main__':
    main()
