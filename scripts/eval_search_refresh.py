"""Paired live-budget/horizon diagnostics on the same public-prefix worlds; no live orders."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

from pokesim.belief.sets import Beliefs
from pokesim.belief.team_catalog import TeamCatalog
from pokesim.config import ROOT
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.evaluation.branch_matches import atomic_json, file_sha, hash_seed, summarize
from pokesim.evaluation.shadow import position
from pokesim.live.advisor import OPPONENT_MODEL
from pokesim.search.lookahead import Lookahead, to_id
from pokesim.search.planner import plan, predict
from pokesim.search.terminal import search
from pokesim.value.runtime import ValueEnsemble

VARIANTS = {'base': (8500, 8), 'more_time': (15000, 8), 'longer_horizon': (15000, 12)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--priors', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    hash_seed()
    args.out.mkdir(parents=True, exist_ok=False)
    value = ValueEnsemble(ROOT / 'runs/value-replays-v1')
    prior = sampler(load_bundle(args.priors))
    prior.catalog = TeamCatalog.load(prior.base_id)
    engine = Lookahead(prior, samples=4, value_model=value)
    sets = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
    beliefs = Beliefs(sets, prior)
    own = {prior.base_id(to_id(s['species'])): s for s in sets}
    report = {'scope': 'development positions; hypothetical sets; public continuations; no human win-rate claim',
              'catalog_sha': prior.catalog.sha, 'variants': VARIANTS, 'cases': []}
    try:
        for case in json.loads(args.cases.read_text()):
            path = args.records / case['file']
            rec = json.loads(path.read_text())
            session, battle = position(rec, sets, case['turn'])
            belief = beliefs.get(path.name, session.log, 'p1')
            settings = copy.deepcopy(rec['versions']['planner']['search'])
            settings.pop('value_model', None)
            settings.update(workers=4, policyNets={'opp': str(ROOT / 'runs/opp-replays-v1/opp-seed0.json')})
            request = engine.query(battle, own, search=settings, belief=belief, log=session.log,
                                   value=True, capture=True)
            atomic_json(args.out / (path.stem + f'-T{case["turn"]}.json'), request)
            last = session.last_move('p2')
            def predictor(payoff, labels, p=None):
                return predict(payoff, labels, OPPONENT_MODEL, last, None, .5, p)
            row = {**case, 'sha256': file_sha(path), 'variants': {}}
            names = list(VARIANTS)
            random.Random(path.name).shuffle(names)
            for name in names:
                budget, horizon = VARIANTS[name]
                query = copy.deepcopy(request)
                query.update(decisionMs=budget, maxTurns=horizon)
                started = time.monotonic()
                matrices = search(engine, query, predictor=predictor)
                advice = plan(matrices, OPPONENT_MODEL, last, pure=True)
                row['variants'][name] = {'chosen': advice.ranked()[0][0], 'seconds': time.monotonic() - started,
                    'search': advice.status['terminal_search'], 'root_evidence': advice.status.get('root_evidence')}
                print(path.name, case['turn'], name, row['variants'][name]['chosen'],
                      row['variants'][name]['search']['simulations'], flush=True)
            actions = sorted({v['chosen'] for v in row['variants'].values()})
            trial = {k: request[k] for k in ['format', 'samples', 'observationLog']}
            trial['branches'] = {'informationScope': 'public_observation', 'observationLog': session.log,
                'seeds': [0, 1], 'maxTurns': 160, 'actions': actions,
                'continuations': ['planned/tactical', 'planned/disrupt', 'planned/expert']}
            engine.query_timeout = 120.
            row['terminal'] = summarize(engine._query(trial))
            report['cases'].append(row)
            atomic_json(args.out / 'report.json', report)
    finally:
        beliefs.close()
        engine.close()


if __name__ == '__main__':
    main()
