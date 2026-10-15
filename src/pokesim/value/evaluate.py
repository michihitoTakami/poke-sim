"""Battle-weighted development diagnostics and paired battle-bootstrap intervals. This is not a final test."""
import argparse
import json
from pathlib import Path

import numpy as np

from pokesim.evaluation.metrics import reliability


def group_means(values, groups, weights):
    keys, indices = np.unique(groups, return_inverse=True)
    numerator = np.bincount(indices, weights=np.asarray(values) * weights)
    denominator = np.bincount(indices, weights=weights)
    return keys, numerator / denominator


def losses(p, target):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return {'brier': (p - target) ** 2,
            'log_loss': -target * np.log(p) - (1 - target) * np.log(1 - p)}


def intervals(values, samples=2000, seed=0):
    # Resample source battles, never individual turns, sampled worlds or seeds.
    rng = np.random.default_rng(seed)
    means = np.asarray(values)[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def score(predictions, data, bootstrap=2000):
    y, w, g = data['target'], data['weight'], data['group']
    out = {'rows': len(y), 'battles': len(set(g)), 'models': {}, 'paired_differences': {}}
    cached = {}
    for name, p in predictions.items():
        if len(p) != len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('invalid value probabilities')
        cached[name] = {metric: group_means(v, g, w)[1] for metric, v in losses(p, y).items()}
        summary = {metric: {'mean': float(v.mean()), 'ci95': intervals(v, bootstrap)}
                   for metric, v in cached[name].items()}
        summary['reliability'] = reliability(p, y, bins=5, w=w)
        out['models'][name] = summary
    pairs = [(a, b) for a in predictions for b in ('constant', 'hp', 'scratch') if b in predictions and a != b
             and a not in ('constant', 'hp')]
    for a, b in pairs:
        out['paired_differences'][f'{a}_minus_{b}'] = {
            metric: {'mean': float((cached[a][metric] - cached[b][metric]).mean()),
                     'ci95': intervals(cached[a][metric] - cached[b][metric], bootstrap)}
            for metric in ('brier', 'log_loss')}
    return out


def report(predictions, data, slice_names, bootstrap=2000):
    result = score(predictions, data, bootstrap)
    result['slices'] = {}
    for index, name in enumerate(slice_names):
        mask = data['flags'][:, index]
        if mask.any():
            sub = {k: v[mask] for k, v in data.items()}
            # Each represented source battle has equal weight within this slice too.
            _, indices = np.unique(sub['group'], return_inverse=True)
            totals = np.bincount(indices, weights=sub['weight'])
            sub['weight'] = sub['weight'] / totals[indices]
            result['slices'][name] = score({k: v[mask] for k, v in predictions.items()}, sub, bootstrap)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / 'report.json').read_text())
    for source in ('public', 'reference'):
        print(source, result[source]['battles'], 'development battles')
        for name, metrics in result[source]['models'].items():
            print(f"  {name:16s} Brier {metrics['brier']['mean']:.5f} log loss {metrics['log_loss']['mean']:.5f}")


if __name__ == '__main__':
    main()
