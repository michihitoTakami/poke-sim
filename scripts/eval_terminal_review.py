"""Validate the surplus-budget terminal review on already frozen development hypotheses."""
import argparse
import json
import time
from pathlib import Path

from pokesim.evaluation.branch_matches import atomic_json
from pokesim.search.lookahead import Lookahead
from pokesim.search.review import preferred, review


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--search', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    source = json.loads((args.search / 'report.json').read_text())
    engine = Lookahead(None)
    report = {'scope': 'development; fixed hypotheses; public policies; 5.5-second surplus allowance', 'cases': []}
    try:
        for case in source['cases']:
            request = json.loads((args.search / (Path(case['file']).stem + f'-T{case["turn"]}.json')).read_text())
            baseline = case['variants']['base']
            labels = [x['label'] for x in baseline['root_evidence']]
            started = time.monotonic()
            result = review(engine, request, [{'p1': labels, 'terminal_search': baseline['search']}], started + 5.5)
            chosen = preferred(result, baseline['chosen'])
            row = {'file': case['file'], 'turn': case['turn'], 'base': baseline['chosen'], 'chosen': chosen,
                   'seconds': time.monotonic() - started, 'review': result,
                   'naive_more_time': case['variants']['more_time']['chosen'],
                   'naive_longer_horizon': case['variants']['longer_horizon']['chosen']}
            report['cases'].append(row)
            atomic_json(args.out, report)
            print(case['file'], case['turn'], baseline['chosen'], '=>', chosen,
                  result.get('complete'), result.get('reason'), row['seconds'], flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
