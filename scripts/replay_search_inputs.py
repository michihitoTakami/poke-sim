"""Re-evaluate frozen observed positions and hidden-set hypotheses without playing or using future events."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from pokesim.live.advisor import OPPONENT_MODEL
from pokesim.search.lookahead import Lookahead
from pokesim.search.planner import plan
from pokesim.search.terminal import search
from pokesim.value.runtime import ValueEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--value-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    text = args.inputs.read_bytes()
    engine = Lookahead(None, value_model=ValueEnsemble(args.value_run), opponent_model=OPPONENT_MODEL)
    report = {'input_sha256': hashlib.sha256(text).hexdigest(), 'decisions': []}
    try:
        for case in json.loads(text):
            results = search(engine, copy.deepcopy(case['request']))
            advice = plan(results, engine.opponent_model, case.get('last_move'), pure=True)
            report['decisions'].append({'source': case['source'], 'turn': case['turn'],
                                        'ranked': advice.ranked(), 'status': advice.status})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(f"{case['source']} T{case['turn']}: {advice.ranked()[0][0]}", flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
