"""Replay frozen, public-prefix search requests with the deployed value model; sends no ladder orders."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from pokesim.evaluation.branch_matches import atomic_json
from pokesim.live.advisor import OPPONENT_MODEL
from pokesim.search.lookahead import Lookahead, to_id
from pokesim.search.planner import plan, predict
from pokesim.search.review import search_with_review
from pokesim.value.runtime import ValueEnsemble


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--value-run', type=Path, required=True)
    ap.add_argument('--opponent-policy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    engine = Lookahead(None, value_model=ValueEnsemble(args.value_run))
    report = {'scope': 'diagnostic public-prefix hypotheses; not an untouched evaluation or a win-rate estimate',
              'value_model': engine.value_model.version, 'cases': []}
    try:
        for path in sorted(args.requests.glob('*-T*.json')):
            request = json.loads(path.read_text())
            request['policyNets'] = {'opp': str(args.opponent_policy.resolve())}
            last = next((to_id(line.split('|')[3]) for line in reversed(request['observationLog'])
                         if line.startswith('|move|p2')), None)

            def predictor(payoff, labels, prior=None):
                return predict(payoff, labels, OPPONENT_MODEL, last, None, .5, prior)

            started = time.monotonic()
            matrices = search_with_review(engine, copy.deepcopy(request), predictor=predictor)
            advice = plan(matrices, OPPONENT_MODEL, last, pure=True)
            row = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'chosen': advice.ranked()[0][0], 'seconds': time.monotonic() - started,
                   'allowance_ms': request.get('decisionMs', 8500), 'status': advice.status}
            report['cases'].append(row)
            atomic_json(args.out, report)
            print(path.name, row['chosen'], f"{row['seconds']:.2f}s",
                  'counterplay', advice.status.get('root_counterplay', {}).get('complete'), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
