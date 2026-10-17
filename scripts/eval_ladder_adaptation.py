"""Compare ladder fine-tuning with frozen incumbents on identical development and held-out observations.

Does not train or select models. Confidence intervals resample battles, not individual turns.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pokesim.evaluation.replay_value import battle_key
from pokesim.learn.model import load_checkpoint
from pokesim.learn.train_policy import evaluate, load
from pokesim.learn.train_value import load as load_value
from pokesim.runtime.bundle import file_hash
from pokesim.value.evaluate import group_means, losses
from pokesim.value.model import predict
from pokesim.value.runtime import ValueEnsemble


def keys(groups):
    return {battle_key(str(g).removesuffix(':p1').removesuffix(':p2')) for g in groups}


def check_scope(data, before, after):
    """Reject fitting overlap with either compared run; development remains used for epoch selection."""
    held = keys(data['group'])
    exposures = {}
    for name, run in (('before', before), ('after', after)):
        report = json.loads((run / 'report.json').read_text())
        provenance = report['provenance']
        for split in ('train', 'pretrain'):
            for source in provenance.get(split, []):
                path = Path(source['path'])
                if file_hash(path) != source['sha']:
                    raise ValueError(f'modified source tensors: {path}')
                with np.load(path, allow_pickle=False) as tensors:
                    if keys(tensors['group']) & held:
                        raise ValueError(f'evaluation battles were used for fitting: {path}')
        exposures[name] = file_hash(run / 'report.json')
    return exposures


def policy_scores(run, data):
    report = json.loads((run / 'report.json').read_text())
    seeds = {}
    for seed, item in report['seeds'].items():
        path = run / item['checkpoint']
        if path.parent != run or file_hash(path) != item['sha']:
            raise ValueError('modified opponent checkpoint')
        model, _ = load_checkpoint(path)
        seeds[seed] = evaluate(model, data, 'cpu')['all']
    return {'seeds': seeds, 'mean': {k: float(np.mean([s[k] for s in seeds.values()]))
                                    for k in ('top1', 'top3', 'log_lik')}}


def value_scores(before, after, data):
    predictions = {}
    for name, run in (('before', before), ('after', after)):
        ensemble = ValueEnsemble(run)
        predictions[name] = np.mean([predict(m, data) for m in ensemble.models], axis=0)
    grouped = {name: {k: group_means(v, data['group'], data['weight'])[1]
                      for k, v in losses(p, data['target']).items()} for name, p in predictions.items()}
    out = {name: {k: float(v.mean()) for k, v in scores.items()} for name, scores in grouped.items()}
    paired = {}
    for metric in ('brier', 'log_loss'):
        delta = grouped['after'][metric] - grouped['before'][metric]
        boot = delta[np.random.default_rng(0).integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
        paired[metric] = {'mean': float(delta.mean()), 'ci90': np.quantile(boot, [.05, .95]).tolist()}
    out['after_minus_before'] = paired
    out['battles'], out['rows'] = len(set(data['group'])), len(data['target'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--value-data', type=Path, required=True)
    ap.add_argument('--value-before', type=Path, required=True)
    ap.add_argument('--value-after', type=Path, required=True)
    ap.add_argument('--opp-before', type=Path, required=True)
    ap.add_argument('--opp-after', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    results = {}
    prior_groups = {}
    for split in ('train', 'development', 'test'):
        with np.load(args.value_data / f'value-{split}.npz') as data:
            prior_groups[split] = keys(data['group'])
    for a, b in (('train', 'development'), ('train', 'test'), ('development', 'test')):
        if prior_groups[a] & prior_groups[b]:
            raise ValueError(f'battle overlap: {a}/{b}')
    for split in ('development', 'test'):
        policy = load(args.data / f'opponent-{split}.npz')
        value = load_value(args.value_data / f'value-{split}.npz')
        check_scope(policy, args.opp_before, args.opp_after)
        check_scope(value, args.value_before, args.value_after)
        results[split] = {'policy': {'before': policy_scores(args.opp_before, policy),
                                     'after': policy_scores(args.opp_after, policy)},
                          'value': value_scores(args.value_before, args.value_after, value)}
    report = {'schema_version': 1, 'settings': {k: str(v) for k, v in vars(args).items()},
              'data_shas': {str(p): file_hash(p) for split in ('train', 'development', 'test') for p in
                            (args.data / f'opponent-{split}.npz', args.value_data / f'value-{split}.npz')},
              'report_shas': {str(p): file_hash(p / 'report.json') for p in
                              (args.value_before, args.value_after, args.opp_before, args.opp_after)},
              'note': 'Development chooses epochs. Test is descriptive held-out validation, six battles; '
                      'all histories were inspected for the weakness audit. No claim of online win-rate gain.',
              'results': results}
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
