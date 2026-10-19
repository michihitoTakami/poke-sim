"""Fixed-prefix, fixed-trial ablation of unconditional boost bonuses; no live orders."""
import argparse
import copy
import json
from pathlib import Path

from eval_relay_tempo import fixed_search

from pokesim.config import ROOT
from pokesim.evaluation.branch_matches import atomic_json, file_sha, hash_seed
from pokesim.search.evaluation import LEGACY_VERSION, VERSION
from pokesim.search.lookahead import Lookahead
from pokesim.search.planner import plan
from pokesim.value.runtime import ValueEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--value-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=256)
    args = parser.parse_args()
    hash_seed()
    if args.out.exists() or args.trials < 1:
        raise ValueError('choose a new output file and positive trial budget')
    value, engine = ValueEnsemble(args.value_run), Lookahead(None)
    engine.query_timeout = 120
    report = dict(scope='paired development prefixes and hypothetical sets; not human win rate',
                  trials=args.trials, value=value.version, cases=[])
    try:
        for path in sorted(args.fixtures.glob('*.request.json')):
            req = json.loads(path.read_text())
            row = dict(case=path.name, source_sha=file_sha(path), variants={})
            for version in (LEGACY_VERSION, VERSION):
                request = copy.deepcopy(req)
                request['valueUtility'] = version
                result = fixed_search(ROOT, request, value, args.trials, utility=version)
                row['variants'][version] = dict(action=plan(result, pure=True).ranked()[0][0],
                                                failures=sum(r['terminal_search']['failed'] for r in result))
            actions = list(dict.fromkeys(v['action'] for v in row['variants'].values()))
            if len(actions) > 1:
                reply = engine._query(dict(format=req['format'], samples=req['samples'][:4], branches=dict(
                    seeds=[0, 1], continuations=['planned/tactical', 'planned/disrupt', 'planned/expert',
                                                 'planned/counterplay'], maxTurns=160,
                    informationScope='public_observation', observationLog=req['observationLog'],
                    livePolicies=True, recordTrajectories=False, actions=actions)))
                row['terminal_comparison'] = {
                    a: {outcome: sum(t['action'] == a and t['outcome'] == outcome for t in reply['trials'])
                        for outcome in ('win', 'loss', 'draw', 'failed', 'unfinished')} for a in actions}
            report['cases'].append(row)
            print(json.dumps(row), flush=True)
        atomic_json(args.out, report)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
