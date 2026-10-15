"""Pretrain public value, then compare fixed-reference fine-tuning with/without pretraining over paired seeds."""
import argparse
import copy
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pokesim.evaluation.branch_matches import atomic_json, digest, file_sha, tree_sha
from pokesim.runtime.bundle import git_commit
from pokesim.value.dataset import load
from pokesim.value.evaluate import group_means, losses, report
from pokesim.value.model import ValueModel, load_checkpoint, predict


def hp_fit(data, penalty=.001):
    x = np.column_stack((np.ones(len(data['target'])), data['baseline'])).astype(np.float64)
    y, w = data['target'], data['weight'].astype(np.float64)
    w /= w.sum()
    beta = np.zeros(x.shape[1])
    ridge = np.eye(len(beta)) * penalty
    ridge[0, 0] = 1e-8
    for _ in range(80):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        gradient = x.T @ (w * (p - y)) + ridge @ beta
        hessian = x.T @ ((w * p * (1 - p))[:, None] * x) + ridge
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def hp_predict(beta, data):
    x = np.column_stack((np.ones(len(data['target'])), data['baseline']))
    return 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(model, train, dev, *, stage, seed, epochs, patience, batch, lr, device, out, contract):
    model = model.to(device)
    private = stage != 'public'
    public_t = torch.as_tensor(train['public'], device=device)
    private_t = torch.as_tensor(train['private'], device=device) if private else None
    target_t = torch.as_tensor(train['target'], device=device)
    weight_t = torch.as_tensor(train['weight'] * (len(train['weight']) / train['weight'].sum()), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, saved, history, idle = float('inf'), 0, None, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(target_t), generator=generator).to(device)
        total = 0.
        for indices in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(public_t[indices], private_t[indices] if private else None)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, target_t[indices], reduction='none') *
                    weight_t[indices]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        p = predict(model, dev, device, private=private)
        value = float(group_means(losses(p, dev['target'])['log_loss'], dev['group'], dev['weight'])[1].mean())
        history.append({'epoch': epoch, 'train_log_loss': total / len(target_t), 'development_log_loss': value})
        print(f'{stage} seed={seed} epoch={epoch}: train={total / len(target_t):.5f} dev={value:.5f}', flush=True)
        if value < best - 1e-6:
            best, best_epoch, idle = value, epoch, 0
            saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            idle += 1
        if idle >= patience:
            break
    model.load_state_dict(saved)
    checkpoint = dict(contract, schema_version=1, stage=stage, seed=seed, best_epoch=best_epoch,
                      model_config=model.config, state_dict=saved, best_development_log_loss=best)
    filename = out / f'{stage}-seed{seed}.pt'
    torch.save(checkpoint, filename)
    # Reload through the inference boundary; selected checkpoint must give the same probabilities.
    reloaded, _ = load_checkpoint(filename, contract['dataset_id'])
    reload_p = predict(reloaded.to(device), dev, device, private=private)
    expected = predict(model, dev, device, private=private)
    if not np.array_equal(reload_p, expected):
        raise ValueError('saved checkpoint does not reproduce model predictions')
    return model.cpu(), {'checkpoint': filename.name, 'sha': file_sha(filename), 'best_epoch': best_epoch,
                         'history': history}


def run(args):
    if args.out.exists():
        raise ValueError('refusing to overwrite an existing value training run')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data, meta = load(args.data)
    args.out.mkdir(parents=True)
    code_dir = Path(__file__).parent
    contract = {'dataset_id': meta['dataset_id'], 'data_manifest_sha': file_sha(args.data / 'manifest.json'),
                'features': meta['features'], 'dex_sha': meta['dex_sha'], 'reference': meta['reference'],
                'code_sha': tree_sha(code_dir, {'.py'}), 'code': git_commit(),
                'torch': str(torch.__version__), 'python': platform.python_version(),
                'device': args.device, 'device_name': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
                'deterministic': True, 'target': 'terminal win=1/loss=0/draw=.5; no discount or HP shaping',
                'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'evaluation': 'development used for epoch selection; final test and live play untouched'}
    atomic_json(args.out / 'manifest.json', contract)
    trained = {'public': [], 'scratch': [], 'pretrained': []}
    stage_results = {}
    public_dim, private_dim = len(meta['features']['public']), len(meta['features']['private'])
    for seed in args.seeds:
        set_seed(seed)
        initial = ValueModel(public_dim, private_dim, args.hidden)
        common = dict(seed=seed, patience=args.patience, batch=args.batch_size,
                      device=args.device, out=args.out, contract=contract)
        model, result = fit(copy.deepcopy(initial), data['public_train'], data['public_development'],
                            stage='public', epochs=args.public_epochs, lr=args.public_lr, **common)
        trained['public'].append(model)
        stage_results[f'public:{seed}'] = result
        # Reset all RNGs, not just minibatch permutations, for the paired fine-tuning runs.
        for stage, start in (('scratch', initial), ('pretrained', model)):
            set_seed(seed)
            fitted, result = fit(copy.deepcopy(start), data['reference_train'], data['reference_development'],
                                 stage=stage, epochs=args.reference_epochs, lr=args.reference_lr, **common)
            trained[stage].append(fitted)
            stage_results[f'{stage}:{seed}'] = result
    result = {'schema_version': 1, 'dataset_id': meta['dataset_id'], 'provenance': contract,
              'stages': stage_results, 'selection': 'all requested seeds equally averaged; no best-seed selection'}
    for source in ('public', 'reference'):
        train, dev = data[f'{source}_train'], data[f'{source}_development']
        base_rate = float(np.average(train['target'], weights=train['weight']))
        hp = hp_fit(train)
        predictions = {'constant': np.full(len(dev['target']), base_rate), 'hp': hp_predict(hp, dev)}
        selected = ('public',) if source == 'public' else ('scratch', 'pretrained', 'transfer')
        per_seed = {}
        for name in selected:
            models = trained['public' if name == 'transfer' else name]
            ps = [predict(m, dev, private=source == 'reference' and name != 'transfer') for m in models]
            predictions[name] = np.mean(ps, axis=0)
            per_seed[name] = {str(seed): {k: float(group_means(v, dev['group'], dev['weight'])[1].mean())
                                        for k, v in losses(p, dev['target']).items()}
                              for seed, p in zip(args.seeds, ps, strict=True)}
        result[source] = report(predictions, dev, meta['features']['slices'], args.bootstrap)
        result[source]['per_seed'] = per_seed
        result[source]['baseline_parameters'] = {'constant': base_rate, 'hp_logistic': hp.tolist()}
        np.savez_compressed(args.out / f'{source}-development-predictions.npz',
                            **predictions, target=dev['target'], weight=dev['weight'], group=dev['group'])
    result['report_id'] = digest(result)
    atomic_json(args.out / 'report.json', result)
    for source in ('public', 'reference'):
        print(source, result[source]['battles'], 'development battles', flush=True)
        for name, metrics in result[source]['models'].items():
            print(f"  {name:16s} Brier {metrics['brier']['mean']:.5f} log loss {metrics['log_loss']['mean']:.5f}")
    return result


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=positive, default=4)
    parser.add_argument('--hidden', type=positive, default=96)
    parser.add_argument('--batch-size', type=positive, default=1024)
    parser.add_argument('--public-epochs', type=positive, default=30)
    parser.add_argument('--reference-epochs', type=positive, default=50)
    parser.add_argument('--patience', type=positive, default=8)
    parser.add_argument('--public-lr', type=float, default=.001)
    parser.add_argument('--reference-lr', type=float, default=.0003)
    parser.add_argument('--bootstrap', type=positive, default=2000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('seeds must be distinct nonnegative integers')
    if args.public_lr <= 0 or args.reference_lr <= 0 or args.hidden < 2:
        parser.error('learning rates must be positive and hidden >= 2')
    run(args)


if __name__ == '__main__':
    main()
