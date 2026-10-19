"""Train a value net on real game outcomes (pokesim.learn.value_dataset) and compare it with the baselines.

  uv run python -m pokesim.learn.train_value --out runs/value-real-v1 \
      --pretrain data/processed/learn/value-v1/replays.npz \
      --train data/processed/learn/value-v1/ladder-train.npz \
      --dev data/processed/learn/value-v1/ladder-development.npz \
      [--reference-run runs/terminal-value-v2] [--seeds 0 1 2] [--device cuda] [--hidden 256]

Per seed: pretrain on the replay examples, fine-tune on the ladder examples, epochs chosen on the battle-weighted
development log loss. The report scores, on the same development rows, the constant rate, the HP/material logistic
baseline, the current live ensemble (--reference-run: V_ref, trained for a reference policy, applied to real
outcomes) and the new ensemble, with battle-bootstrap intervals and paired differences, plus a reliability
table. Gate C (offline): battle-mean Brier at least 0.015 below the better of the HP baseline and the live
ensemble, with the paired difference's 90% interval excluding 0.
"""
import argparse
import json
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pokesim.data.dataset import sidecar
from pokesim.evaluation.metrics import reliability
from pokesim.runtime.bundle import file_hash, git_commit
from pokesim.value.evaluate import group_means, intervals, losses
from pokesim.value.model import create_model, load_checkpoint, predict
from pokesim.value.train import hp_fit, hp_predict, set_seed


def load(path: Path) -> dict:
    data = dict(np.load(path, allow_pickle=False))
    data['meta'] = json.loads(sidecar(path).read_text())
    return data


def concat(paths):
    parts = [load(Path(p)) for p in paths]
    keys = [k for k in parts[0] if k != 'meta']
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    data['group'] = battle_groups(data['group'])
    _, inverse = np.unique(data['group'], return_inverse=True)
    data['weight'] = data['weight'] / np.bincount(inverse, weights=data['weight'])[inverse]
    data['weight'] = data['weight'] / data['weight'].sum() * len(data['weight'])
    data['meta'] = [p['meta'] for p in parts]
    return data


def battle_groups(groups):
    """Both public replay actors, and all their turns, belong to one statistical unit."""
    return np.array([re.sub(r':p[12]$', '', str(g)) for g in groups])


def dev_log_loss(model, dev, device) -> float:
    p = predict(model, dev, device)
    return float(group_means(losses(p, dev['target'])['log_loss'], dev['group'], dev['weight'])[1].mean())


def fit(model, train, dev, *, epochs, lr, patience, batch, seed, device, log) -> dict:
    model = model.to(device)
    public = torch.as_tensor(train['public'], device=device)
    private = torch.as_tensor(train['private'], device=device)
    target = torch.as_tensor(train['target'], device=device)
    weight = torch.as_tensor(train['weight'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, idle = dev_log_loss(model, dev, device), 0, 0
    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=generator).to(device)
        total = 0.
        for indices in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(public[indices], private[indices])
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, target[indices], reduction='none') *
                    weight[indices]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        value = dev_log_loss(model, dev, device)
        log(f'  epoch {epoch}: train {total / len(target):.5f} dev {value:.5f}')
        if value < best - 1e-6:
            best, best_epoch, idle = value, epoch, 0
            saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            idle += 1
            if idle >= patience:
                break
    model.load_state_dict(saved)
    return {'best_epoch': best_epoch, 'best_dev_log_loss': best}


def reference_predictions(run: Path, dev: dict) -> np.ndarray | None:
    from pokesim.value.runtime import ValueEnsemble

    ensemble = ValueEnsemble(run)
    return np.mean([predict(m, dev) for m in ensemble.models], axis=0)


def main() -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pretrain', nargs='*', default=[])
    ap.add_argument('--init-run', type=Path, help='fine-tune matching seeds from an existing real-outcome run')
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--dev', nargs='+', required=True)
    ap.add_argument('--reference-run', default=None)
    ap.add_argument('--reference-dev', type=Path, help='aligned development tensors for a different incumbent encoder')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--pretrain-epochs', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--pretrain-lr', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    if args.init_run and args.pretrain:
        raise ValueError('choose --init-run or --pretrain')
    out = Path(args.out)
    if out.exists():
        raise ValueError('choose a new run directory')
    out.mkdir(parents=True)
    log_file = open(out / 'training.log', 'w')

    def log(text):
        print(text, flush=True)
        log_file.write(text + '\n')
        log_file.flush()
    train, dev = concat(args.train), concat(args.dev)
    pretrain = concat(args.pretrain) if args.pretrain else None
    metas = [m for d in (train, dev, pretrain) if d for m in d['meta']]
    if len({m['feature_code_sha'] for m in metas}) != 1 or len({m['dex_sha'] for m in metas}) != 1:
        raise ValueError('tensors were built with different feature code or dex')
    if set(train['group']) & set(dev['group']) or (
            pretrain is not None and set(pretrain['group']) & set(dev['group'])):
        raise ValueError('a battle appears in both training and development data')
    device = torch.device(args.device)
    features = train['meta'][0]['features']
    initial = None
    if args.init_run:
        initial = json.loads((args.init_run / 'report.json').read_text())
        cfg = initial['provenance']
        if (cfg.get('target') != 'real_outcome' or cfg['features'] != features or
                cfg['feature_code_sha'] != train['meta'][0]['feature_code_sha'] or
                cfg['dex_sha'] != train['meta'][0]['dex_sha']):
            raise ValueError('initial value run uses a different target or feature contract')
    contract = {'schema_version': 1, 'target': 'real_outcome', 'feature_version': train['meta'][0]['feature_version'],
                'features': features, 'feature_code_sha': train['meta'][0]['feature_code_sha'],
                'dex_sha': train['meta'][0]['dex_sha'], 'code': git_commit(), 'torch': str(torch.__version__),
                'training_code_sha': file_hash(Path(__file__)),
                'model_code_sha': {name: file_hash(Path(__file__).parents[1] / 'value' / name)
                                   for name in ('model.py', 'set_model.py')},
                'grouping': 'underlying battle; replay :p1/:p2 perspectives kept together; total battle weight=1',
                'deterministic': True,
                'python': platform.python_version(), 'device': str(device),
                'train': [{'path': p, 'sha': file_hash(Path(p))} for p in args.train],
                'dev': [{'path': p, 'sha': file_hash(Path(p))} for p in args.dev],
                'pretrain': [{'path': p, 'sha': file_hash(Path(p))} for p in args.pretrain],
                'settings': {**vars(args), 'init_run': str(args.init_run) if args.init_run else None,
                             'reference_dev': str(args.reference_dev) if args.reference_dev else None},
                'initialization': {'run': str(args.init_run),
                                   'report_sha': file_hash(args.init_run / 'report.json'),
                                   'checkpoints': {}} if args.init_run else None}
    (out / 'manifest.json').write_text(json.dumps(contract, indent=1))
    started = time.perf_counter()
    models, stages = [], {}
    for seed in args.seeds:
        set_seed(seed)
        model = create_model(features, args.hidden)
        log(f'seed {seed}')
        result = {}
        if initial:
            item = initial['stages'][f'real:{seed}']
            path = args.init_run / item['checkpoint']
            if path.parent != args.init_run or file_hash(path) != item['sha']:
                raise ValueError('modified initial value checkpoint')
            model, saved = load_checkpoint(path)
            if saved['features'] != features or saved.get('stage') != 'real':
                raise ValueError('initial value checkpoint contract mismatch')
            contract['initialization']['checkpoints'][str(seed)] = item['sha']
            result['initial'] = {'dev_log_loss': dev_log_loss(model.to(device), dev, device)}
        if pretrain is not None:
            log(' pretrain')
            result['pretrain'] = fit(model, pretrain, dev, epochs=args.pretrain_epochs, lr=args.pretrain_lr,
                                     patience=args.patience, batch=args.batch, seed=seed, device=device, log=log)
        log(' fine-tune')
        result['finetune'] = fit(model, train, dev, epochs=args.epochs, lr=args.lr, patience=args.patience,
                                 batch=args.batch, seed=seed, device=device, log=log)
        model = model.cpu()
        path = out / f'real-seed{seed}.pt'
        torch.save({'schema_version': 1, 'dataset_id': contract['feature_code_sha'], 'stage': 'real',
                    'seed': seed, 'model_config': model.config, 'state_dict': model.state_dict(),
                    'features': features, 'reference': {'target': 'real_outcome'}, **result}, path)
        stages[f'real:{seed}'] = {'checkpoint': path.name, 'sha': file_hash(path), **result}
        models.append(model)
    rate = float(np.average(train['target'], weights=train['weight']))
    predictions = {'constant': np.full(len(dev['target']), rate),
                   'hp': hp_predict(hp_fit(train), dev),
                   'real': np.mean([predict(m, dev) for m in models], axis=0)}
    if args.reference_run:
        reference_dev = concat([args.reference_dev]) if args.reference_dev else dev
        if any(not np.array_equal(reference_dev[k], dev[k]) for k in ('group', 'turn', 'target')):
            raise ValueError('incumbent development rows are not aligned with candidate rows')
        predictions['reference_v2'] = reference_predictions(Path(args.reference_run), reference_dev)
    y, w, g = dev['target'], dev['weight'], dev['group']
    cached = {name: {k: group_means(v, g, w)[1] for k, v in losses(p, y).items()} for name, p in predictions.items()}
    report = {'schema_version': 1, 'provenance': contract, 'stages': stages, 'dev_rows': int(len(y)),
              'dev_battles': int(len(set(g))), 'models': {}, 'paired_differences': {}, 'by_turn': {}}
    for name, p in predictions.items():
        report['models'][name] = {k: {'mean': float(v.mean()), 'ci95': intervals(v, args.bootstrap)}
                                  for k, v in cached[name].items()}
        report['models'][name]['reliability'] = reliability(p, y, bins=5, w=w)
    for other in ('hp', 'reference_v2', 'constant'):
        if other in cached:
            d = {k: cached['real'][k] - cached[other][k] for k in ('brier', 'log_loss')}
            report['paired_differences'][f'real_minus_{other}'] = {
                k: {'mean': float(v.mean()), 'ci90': [float(x) for x in np.quantile(
                    np.asarray(v)[np.random.default_rng(0).integers(0, len(v), (args.bootstrap, len(v)))].mean(axis=1),
                    [.05, .95])]} for k, v in d.items()}
    for name, lo, hi in (('early', 0, 3), ('middle', 3, 8), ('late', 8, 10 ** 6)):
        mask = (dev['turn'] > lo) & (dev['turn'] <= hi)
        if mask.any():
            report['by_turn'][name] = {m: float(losses(p[mask], y[mask])['brier'].mean())
                                       for m, p in predictions.items()}
    best_name = min((k for k in ('hp', 'reference_v2') if k in report['models']),
                    key=lambda k: report['models'][k]['brier']['mean'])
    best_other = report['models'][best_name]['brier']['mean']
    gap = best_other - report['models']['real']['brier']['mean']
    paired = report['paired_differences'][f'real_minus_{best_name}']
    report['gate'] = {'brier_gain': float(gap), 'required': .015, 'ci90_excludes_zero': paired['brier']['ci90'][1] < 0,
                      'passed': gap >= .015 and paired['brier']['ci90'][1] < 0}
    report['seconds'] = time.perf_counter() - started
    (out / 'manifest.json').write_text(json.dumps(contract, indent=1))
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    for name, m in report['models'].items():
        log(f"  {name:14s} Brier {m['brier']['mean']:.5f} log loss {m['log_loss']['mean']:.5f}")
    log(json.dumps(report['gate']))
    log_file.close()


if __name__ == '__main__':
    main()
