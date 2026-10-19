"""Compare frozen value candidates on saved exact requests and public turn-start prefixes.

--after freezes a prospective cohort boundary; --code isolates a deployment. Existing inspected games
remain diagnostics. This measures outcome prediction under recorded play, not counterfactual action quality,
and never marks a candidate adopted. Learning and final policy A/B gates remain separate.
"""
import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.evaluation.branch_matches import atomic_json, file_sha
from pokesim.value.evaluate import group_means, losses
from pokesim.value.runtime import ValueEnsemble


def clustered_difference(delta, groups, opponents, samples=5000):
    keys, means = group_means(delta, np.asarray(groups), np.ones(len(groups)))
    labels = [re.sub('[^a-z0-9]', '', str(opponents[k]).lower()) for k in keys]
    if any(not label for label in labels):
        raise ValueError('opponent identity missing; cannot claim opponent-clustered uncertainty')
    _, inverse = np.unique(labels, return_inverse=True)
    totals, counts = np.bincount(inverse, weights=means), np.bincount(inverse)
    indices = np.random.default_rng(719).integers(0, len(totals), (samples, len(totals)))
    draws = totals[indices].sum(1) / counts[indices].sum(1)
    return dict(mean=float(means.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                battles=len(keys), opponent_groups=len(totals))


def observed_rows(records, after='', code=None):
    process = subprocess.Popen(['node', str(ROOT / 'services/sim-tools/record_value_views.js'), str(SHOWDOWN_DIR)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    rows, sources, skipped, exclusions, seen = [], {}, [], Counter(), set()
    try:
        for path in sorted(records.glob('*.json')):
            rec = json.loads(path.read_text())
            if rec.get('started', '') <= after:
                exclusions['before_boundary'] += 1
                continue
            if code and not rec.get('versions', {}).get('code', {}).get('commit', '').startswith(code):
                exclusions['different_code'] += 1
                continue
            bid = rec.get('source')
            if rec.get('result') not in ('win', 'loss', 'draw') or not bid or bid in seen:
                exclusions['duplicate_or_nonterminal'] += 1
                continue
            seen.add(bid)
            process.stdin.write(json.dumps(rec) + '\n')
            process.stdin.flush()
            result = json.loads(process.stdout.readline())
            if result.get('error'):
                raise ValueError(result['error'])
            target = {'win': 1., 'loss': 0., 'draw': .5}[rec['result']]
            rows.extend(dict(r, group=bid, opponent=rec.get('opponent'), target=target) for r in result['rows'])
            skipped.extend(dict(r, file=path.name) for r in result['skipped'])
            sources[path.name] = file_sha(path)
        process.stdin.close()
        if process.wait(timeout=10):
            raise RuntimeError('value prefix worker failed')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    return rows, dict(sources=sources, skipped=skipped, exclusions=dict(exclusions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    parser.add_argument('--after', default='', help='exclusive started timestamp; inspected data are not final tests')
    parser.add_argument('--code', help='deployed commit prefix')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('choose a new report; never overwrite a prospective evaluation')
    rows, provenance = observed_rows(args.records, args.after, args.code)
    if not rows:
        raise ValueError('no eligible normal decisions in the frozen cohort')
    targets, groups = np.array([r['target'] for r in rows]), np.array([r['group'] for r in rows])
    opponents = {r['group']: r['opponent'] for r in rows}
    report = dict(scope='recorded-policy outcome prediction; no action-quality or adoption claim',
                  after=args.after, code=args.code, rows=len(rows), provenance=provenance, models={}, differences={})
    cached = {}
    for label, path in [('incumbent', args.incumbent), ('candidate', args.candidate)]:
        model = ValueEnsemble(path, allow_candidate=label == 'candidate')
        predictions = np.concatenate([model.probabilities([r['view'] for r in rows[i:i + 256]])
                                      for i in range(0, len(rows), 256)])
        cached[label] = losses(predictions, targets)
        report['models'][label] = dict(run=str(path.resolve()), version=model.version, **{
            metric: float(group_means(v, groups, np.ones(len(rows)))[1].mean())
            for metric, v in cached[label].items()})
    for metric in ('brier', 'log_loss'):
        report['differences'][metric] = clustered_difference(
            cached['candidate'][metric] - cached['incumbent'][metric], groups, opponents)
    atomic_json(args.out, report)
    print(json.dumps({k: report[k] for k in ('rows', 'models', 'differences')}, indent=2))


if __name__ == '__main__':
    main()
