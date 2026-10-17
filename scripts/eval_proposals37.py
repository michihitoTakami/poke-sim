"""Paired fixed-world #37 ablations and terminal scoring; simulator diagnostics, never human win rates."""
import argparse
import copy
import json
import time
from pathlib import Path

from pokesim.belief.sets import Beliefs
from pokesim.config import ROOT
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.evaluation.branch_matches import atomic_json, file_sha, summarize, tree_sha
from pokesim.evaluation.shadow import position
from pokesim.search.lookahead import Lookahead, to_id
from pokesim.search.planner import plan
from pokesim.search.terminal import search
from pokesim.value.runtime import ValueEnsemble

CASES = [('20261018-021112', 4), ('20261018-020620', 5), ('20261018-020849', 7),
         ('20261018-021333', 2), ('20261018-021333', 5), ('20261018-012545', 8)]
VARIANTS = {
    'wide-hp': {'rootProgress': True, 'focusTopTwo': False, 'tie_margin': .1},
    'wide-no-hp': {'rootProgress': False, 'focusTopTwo': False, 'tie_margin': .1},
    'strict': {'rootProgress': False, 'focusTopTwo': False, 'tie_margin': .005},
    'narrow': {'rootProgress': False, 'focusTopTwo': False, 'tie_margin': .03},
    'focused': {'rootProgress': False, 'focusTopTwo': True, 'tie_margin': .03},
    'focused-plans': {'rootProgress': False, 'focusTopTwo': True, 'tie_margin': .03},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--priors', type=Path, required=True)
    ap.add_argument('--value-run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--budget-ms', type=int, default=2500)
    ap.add_argument('--worlds', type=int, default=2)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('choose a new output directory')
    args.out.mkdir(parents=True)
    sets = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
    engine = Lookahead(sampler(load_bundle(args.priors)), samples=args.worlds,
                       value_model=ValueEnsemble(args.value_run))
    engine.query_timeout = 300.
    beliefs = Beliefs(sets, engine.sampler)
    own = {engine.sampler.base_id(to_id(s['species'])): s for s in sets}
    rows = []
    worker_sha = tree_sha(ROOT / "services/sim-tools", {".js"})
    python_sha = tree_sha(ROOT / "src/pokesim", {".py"})
    try:
        for stamp, turn in CASES:
            path = next(args.records.glob(stamp + '*.json'))
            rec = json.loads(path.read_text())
            session, battle = position(rec, sets, turn)
            belief = beliefs.get(path.name, session.log, 'p1')
            request = engine.query(battle, own, search={'terminalSearch': True}, belief=belief,
                                   log=session.log, value=True, capture=True)
            request.pop('value_model', None)
            result = {'record': path.name, 'sha': file_sha(path), 'turn': turn,
                      'scope': 'inspected development; sampled sets; same observed prefix/worlds',
                      'worker_code_sha': worker_sha, 'python_code_sha': python_sha,
                      'prior_sha': file_sha(args.priors), 'value_model': engine.value_model.version, 'variants': {}}
            atomic_json(args.out / f'{stamp}-T{turn}.request.json', request)
            for name, settings in VARIANTS.items():
                cfg = dict(settings)
                margin = cfg.pop('tie_margin')
                policy = ['planned'] if name == 'focused-plans' else ['expert']
                req = dict(copy.deepcopy(request), decisionMs=args.budget_ms, workers=1, maxTurns=8,
                           priorVisits=4, rolloutPolicies={'p1': policy, 'p2': ['disrupt', 'tactical', 'expert']},
                           **cfg)
                started = time.monotonic()
                matrices = search(engine, req)
                advice = plan(matrices, pure=True, tie_margin=margin)
                label = advice.ranked()[0][0]
                result['variants'][name] = {'chosen': label, 'elapsed': time.monotonic() - started,
                                            'evidence': advice.status.get('root_evidence'),
                                            'search': advice.status['terminal_search']}
                print(path.name, turn, name, label, flush=True)
            # All choices, including old HP choices, get the same terminal public continuations/chance streams.
            trial_request = {k: request[k] for k in ('format', 'samples', 'observationLog')}
            trial_request['branches'] = {'informationScope': 'public_observation', 'observationLog': session.log,
                'seeds': [0, 1], 'maxTurns': 160,
                'continuations': ['planned/tactical', 'planned/disrupt', 'planned/expert']}
            branches = engine._query(trial_request)
            result['terminal'] = summarize(branches)
            for variant in result['variants'].values():
                variant['terminal'] = result['terminal'][variant['chosen']]
            rows.append(result)
            atomic_json(args.out / 'report.json', rows)
    finally:
        beliefs.close()
        engine.close()


if __name__ == '__main__':
    main()
