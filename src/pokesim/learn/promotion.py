"""Require complete paired policy crossplay, in addition to ranking and value calibration, for promotion."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pokesim.evaluation.branch_matches import atomic_json, file_sha, load_result
from pokesim.learn.branches import paired_interval


def root_choice(result, key):
    choices = {d.get(key) for t in result['trials'] for d in t['trajectory']
               if d['side'] == 'p1' and d['rootIntervention']}
    if len(choices) != 1 or None in choices:
        raise ValueError('root recommendation differs across information-identical worlds')
    return next(iter(choices))


def compare(before, after):
    old_manifest = json.loads((before / 'calibration/manifest.json').read_text())
    new_manifest = json.loads((after / 'calibration/manifest.json').read_text())
    old_cfg, new_cfg = old_manifest['contract'], new_manifest['contract']
    for key in ['positions', 'world_seed', 'worlds', 'seeds', 'continuations', 'analysis_groups',
                'records', 'prior_sha', 'worker_code_sha']:
        if old_cfg[key] != new_cfg[key]:
            raise ValueError('crossplay does not share fixed worlds/continuations/source groups')
    cases, missing = [], []
    for case in old_cfg['positions']:
        old_path = before / 'calibration/positions' / f"{case['id']}.json"
        new_path = after / 'calibration/positions' / f"{case['id']}.json"
        if not old_path.exists() or not new_path.exists():
            missing.append(case['id'])
            continue
        old = load_result(old_path, old_manifest, case)
        new = load_result(new_path, new_manifest, case)
        if old['simulation_request']['samples'] != new['simulation_request']['samples']:
            raise ValueError('crossplay sampled states differ')
        try:
            a = root_choice(old['result'], 'referenceRecommendation')
            b = root_choice(new['result'], 'policyRecommendation')
        except ValueError:
            missing.append(case['id'])
            continue
        cells = []
        for row, action in [(old, a), (new, b)]:
            trials = [t for t in row['result']['trials'] if t['action'] == action]
            if not trials or any(t['target'] is None for t in trials):
                cells = []
                break
            cells.append({(t['world'], t['seed'], t['continuation']): t['target'] for t in trials})
        if len(cells) != 2:
            missing.append(case['id'])
            continue
        if cells[0].keys() != cells[1].keys():
            raise ValueError('missing paired crossplay cell')
        cases.append({'case': case, 'group': old_cfg['analysis_groups'][case['file']],
                      'reference_action': a, 'learned_action': b, 'reference_mean': np.mean(list(cells[0].values())),
                      'learned_mean': np.mean(list(cells[1].values())),
                      'delta': np.mean([cells[1][k] - v for k, v in cells[0].items()]), 'trials': len(cells[0])})
    grouped = defaultdict(list)
    for c in cases:
        grouped[c['group']].append(c['delta'])
    mean = float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else 0.
    interval = paired_interval(np.array([c['delta'] for c in cases]), [c['group'] for c in cases])
    passed = len(grouped) >= 20 and not missing and mean >= .03 and interval[0] > 0
    return {'passed': bool(passed), 'groups': len(grouped), 'mean_delta': mean, 'interval90': interval,
            'minimum_groups': 20, 'minimum_mean_gain': .03, 'incomplete_cases': missing, 'cases': cases,
            'scope': 'paired historical grouped simulator holdout; reference vs fixed learned actor; not live Elo'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.run / 'report.json'
    report = json.loads(report_path.read_text())
    if report['provenance'].get('grouped_curriculum_sha') is None:
        raise ValueError('promotion requires a grouped curriculum')
    if file_sha(args.run / 'ours.json') != report['provenance']['reference']['policy_sha']:
        raise ValueError('fixed policy changed')
    if report['gate'].get('policy_crossplay') is not None:
        raise ValueError('calibration was already opened; do not repeatedly promote a tuned candidate')
    result = compare(args.before, args.after)
    report['gate']['policy_crossplay'] = result
    report['gate']['passed'] = bool(report['gate'].get('statistical_passed') and result['passed'])
    report['gate']['pending_policy_crossplay'] = False
    atomic_json(report_path, report)
    print(json.dumps({k: v for k, v in report['gate'].items() if k != 'policy_crossplay'} |
                     {'policy_crossplay': {k: v for k, v in result.items() if k != 'cases'}}, indent=2))


if __name__ == '__main__':
    main()
