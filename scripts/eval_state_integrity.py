"""Replay fixed observed-prefix decisions using the selected source tree; no live orders.

Use PYTHONPATH=<source>/src to compare source trees. --compare compares both actions under the current tree's
corrected sampled worlds, with paired seeds and terminal continuations. This is a simulator diagnostic.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from pokesim.config import ROOT
from pokesim.evaluation.shadow import position
from pokesim.live.advisor import Advisor
from pokesim.search.lookahead import SetSampler
from pokesim.search.planner import plan, predict
from pokesim.search.terminal import search


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--value-run', type=Path, default=ROOT / 'runs/value-replays-v1')
    ap.add_argument('--opponent-policy', default=f'net:{ROOT}/runs/opp-replays-v1/opp-seed0.json')
    ap.add_argument('--decision-ms', type=int, default=4500)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--compare', type=Path, help='before report; compare the chosen actions in corrected worlds')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    before = json.loads(args.compare.read_text())['decisions'] if args.compare else []
    sets = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
    advisor = Advisor(sets, planner={'mode': 'on'}, policy=None, belief=True, value_run=args.value_run,
                      opponent_policy=args.opponent_policy, sampler=SetSampler(['M-6'], observed={}))
    report = {'code_root': str(ROOT), 'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              'value_model': advisor.engine.value_model.version,
              'opponent_model': advisor.opponent_policy_version,
              'settings': {'decision_ms': args.decision_ms, 'workers': args.workers, 'live_record_prior': False},
              'decisions': [], 'scope': 'Observed-prefix replay and hypothetical-set paired terminal '
                                      'continuations. Development diagnostics, not a human win-rate estimate.'}
    try:
        for index, case in enumerate(cases):
            path = args.records / case['record']
            rec = json.loads(path.read_text())
            session, battle = position(rec, sets, case['turn'])
            belief = advisor.beliefs.get(path.name, session.log, 'p1')
            req = advisor.engine.query(battle, advisor.own_sets, search=advisor.planner['search'], belief=belief,
                                       log=session.log, value=True, capture=True)
            req.pop('value_model', None)
            req.update(decisionMs=args.decision_ms, workers=args.workers)
            (args.out / f'{index:02d}.json').write_text(json.dumps(req))
            last = session.last_move('p2')
            results = search(advisor.engine, copy.deepcopy(req), predictor=lambda m, labels, prior=None:
                             predict(m, labels, advisor.engine.opponent_model, last, None, .5, prior))
            advice = plan(results, advisor.engine.opponent_model, last_move=last, pure=True)
            row = {**case, 'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'request_sha256': hashlib.sha256(json.dumps(req).encode()).hexdigest(),
                   'chosen': advice.ranked()[0][0], 'status': advice.status}
            if before:
                previous = next(r for r in before if (r['record'], r['turn']) == (case['record'], case['turn']))
                row['before'] = previous['chosen']
                if row['before'] != row['chosen']:
                    actions = [row['before'], row['chosen']]
                    reply = advisor.engine._query({'format': req['format'], 'samples': req['samples'][:3],
                        'branches': {'seeds': [0, 1, 2, 3], 'continuations': ['expert/tactical', 'expert/disrupt'],
                                     'maxTurns': 100, 'actions': actions}})
                    row['paired_continuations'] = {a: dict(Counter(t['outcome'] for t in reply['trials']
                                                                  if t['action'] == a)) for a in actions}
            report['decisions'].append(row)
            (args.out / 'report.json').write_text(json.dumps(report, indent=1))
            print(case['record'], case['turn'], row.get('before'), '->', row['chosen'], flush=True)
    finally:
        advisor.close()


if __name__ == '__main__':
    main()
