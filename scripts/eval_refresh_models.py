"""Score frozen refreshed checkpoints once on the untouched opponent-group calibration cohort."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pokesim.learn.model import load_checkpoint
from pokesim.learn.train_policy import evaluate, load, tensors
from pokesim.learn.train_value import concat, reference_predictions
from pokesim.value.evaluate import group_means, losses
from pokesim.value.train import hp_fit, hp_predict


def paired(delta, groups, opponents):
    keys, means = group_means(delta, groups, np.ones(len(delta)))
    labels = [opponents[g] for g in keys]
    _, inverse = np.unique(labels, return_inverse=True)
    totals = np.bincount(inverse, weights=means)
    counts = np.bincount(inverse)
    indices = np.random.default_rng(0).integers(0, len(totals), (4000, len(totals)))
    draws = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {'mean': float(means.mean()), 'ci90': np.quantile(draws, [.05, .95]).tolist(),
            'battles': len(keys), 'opponent_groups': len(totals)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--records', type=Path, required=True)
    ap.add_argument('--value-old', type=Path, required=True)
    ap.add_argument('--value-new', type=Path, required=True)
    ap.add_argument('--opp-old', type=Path, required=True)
    ap.add_argument('--opp-new', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('calibration result already exists; do not tune and rescore the same cohort')
    torch.set_num_threads(2)
    spec = json.loads(args.split.read_text())
    opponents = {}
    for name, split in spec['assignments'].items():
        if split == 'calibration':
            rec = json.loads((args.records / name).read_text())
            opponents[rec['source']] = spec['opponent_groups'][name]
            opponents[rec['source'].split(':')[-1]] = spec['opponent_groups'][name]
    train = concat([str(args.data / 'value-train.npz')])
    held = concat([str(args.data / 'value-calibration.npz')])
    predictions = {name: reference_predictions(run, held) for name, run in
                   [('old', args.value_old), ('new', args.value_new)]}
    predictions['hp'] = hp_predict(hp_fit(train), held)
    value = {name: {k: float(group_means(v, held['group'], held['weight'])[1].mean())
                    for k, v in losses(p, held['target']).items()} for name, p in predictions.items()}
    differences = {name: paired((predictions['new'] - held['target']) ** 2 -
                                (predictions[name] - held['target']) ** 2, held['group'], opponents)
                   for name in ['old', 'hp']}
    value_gate = all(d['mean'] <= -.015 and d['ci90'][1] < 0 for d in differences.values())
    report = {'scope': 'untouched grouped calibration, fixed checkpoint and epoch choices',
              'value': value, 'value_brier_differences': differences,
              'value_adopt': bool(value_gate), 'value_threshold': 'Brier gain >= .015 against old and HP; CI90 < 0'}
    data = load(args.data / 'opp-calibration.npz')
    state, actions, mask, target = tensors(data, 'cpu')
    metrics, logp = {}, {}
    for name, run in [('old', args.opp_old), ('new', args.opp_new)]:
        # Seed 0 is the deployed policy; do not select the best seed using calibration.
        source = json.loads((run / 'report.json').read_text())['seeds']['0']['checkpoint']
        model, _ = load_checkpoint(run / source)
        metrics[name] = evaluate(model, data, 'cpu')['all']
        with torch.no_grad():
            p = torch.log_softmax(model(state, actions, mask), dim=-1).gather(1, target[:, None]).squeeze(1)
        logp[name] = np.maximum(p.numpy(), np.log(1e-3))
    delta = paired(-logp['new'] + logp['old'], data['group'], opponents)
    report.update(opponent=metrics, opponent_log_loss_difference=delta,
                  opponent_adopt=bool(delta['mean'] <= -.02 and delta['ci90'][1] < 0 and
                                      metrics['new']['top1'] >= max(.46, metrics['old']['top1'] - .02) and
                                      metrics['new']['top3'] >= .72))
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
