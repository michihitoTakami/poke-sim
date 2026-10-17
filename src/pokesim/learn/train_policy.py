"""Train a public policy net on view rows: the opponent's policy (what real players do) or ours (the search).

  uv run python -m pokesim.learn.train_policy --head opp --out runs/opp-policy-v1 \
      --pretrain data/processed/learn/features-v1/replays.npz \
      --train data/processed/learn/features-v1/ladder-train-p2.npz \
      --dev data/processed/learn/features-v1/ladder-development-p2.npz [--seeds 0 1 2] [--device cuda]

Stages per seed: pretrain on the replay examples (both sides of public games), then fine-tune on the training
rows; epochs are chosen on the development log loss (battle-weighted). Example weights: rating
clip((r - 1000) / 300, .3, 1.5) when known, divided by the number of that opponent's battles beyond 12, and
normalised so each battle weighs 1. The report records data hashes, the feature contract and the metrics the
gate reads: top-1 / top-3 accuracy and mean log likelihood on the development rows, overall and by rating band.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pokesim.data.dataset import sidecar
from pokesim.learn.model import PolicyNet, export_json, load_checkpoint, save_checkpoint
from pokesim.runtime.bundle import file_hash, git_commit

MAX_OPPONENT_BATTLES = 12


def load(path: Path) -> dict:
    data = dict(np.load(path, allow_pickle=False))
    meta = json.loads(sidecar(path).read_text())
    data['meta'] = meta
    return data


def weights(data: dict, rating: bool = True) -> np.ndarray:
    w = np.ones(len(data['target']), np.float64)
    if rating:
        r = data['rating']
        known = r > 0
        w[known] = np.clip((r[known] - 1000) / 300, .3, 1.5)
    groups, counts = np.unique(data['group'], return_counts=True)
    per_group = dict(zip(groups, counts))
    w /= np.array([per_group[g] for g in data['group']], np.float64)  # each battle weighs one
    opp_battles = {}
    for g, o in set(zip(data['group'], data['opponent'])):
        opp_battles[o] = opp_battles.get(o, 0) + 1
    w /= np.array([max(1., opp_battles.get(o, 1) / MAX_OPPONENT_BATTLES) for o in data['opponent']])
    return (w / w.mean()).astype(np.float32)


def tensors(data: dict, device):
    return (torch.as_tensor(data['state'], device=device), torch.as_tensor(data['actions'], device=device),
            torch.as_tensor(data['mask'], device=device),
            torch.as_tensor(data['target'], device=device, dtype=torch.long))


def evaluate(model: PolicyNet, data: dict, device, batch: int = 4096) -> dict:
    model.eval()
    state, actions, mask, target = tensors(data, device)
    w = weights(data, rating=False)
    logp_all, top1, top3 = [], [], []
    with torch.no_grad():
        for i in range(0, len(target), batch):
            logits = model(state[i:i + batch], actions[i:i + batch], mask[i:i + batch])
            logp = torch.log_softmax(logits, dim=-1)
            t = target[i:i + batch]
            logp_all.append(logp.gather(1, t[:, None]).squeeze(1).cpu().numpy())
            order = logits.argsort(dim=-1, descending=True)
            top1.append((order[:, 0] == t).cpu().numpy())
            top3.append((order[:, :3] == t[:, None]).any(dim=1).cpu().numpy())
    logp_all, top1, top3 = (np.concatenate(x) for x in (logp_all, top1, top3))
    logp_all = np.maximum(logp_all, np.log(1e-3))

    def summary(idx):
        if not len(idx):
            return None
        ww = w[idx] / w[idx].sum()
        return {'rows': int(len(idx)), 'battles': int(len(set(data['group'][idx]))),
                'log_lik': float((logp_all[idx] * ww).sum()), 'top1': float((top1[idx] * ww).sum()),
                'top3': float((top3[idx] * ww).sum())}
    out = {'all': summary(np.arange(len(target)))}
    r = data['rating']
    for name, lo, hi in (('rating<1100', 1, 1100), ('rating1100-1200', 1100, 1200), ('rating>=1200', 1200, 10 ** 6)):
        out[name] = summary(np.flatnonzero((r >= lo) & (r < hi)))
    out['turn1'] = summary(np.flatnonzero(data['turn'] == 1))
    out['later'] = summary(np.flatnonzero(data['turn'] > 1))
    return out


def fit(model: PolicyNet, train: dict, dev: dict, device, epochs: int, lr: float, patience: int, batch: int,
        seed: int, log) -> tuple[dict, int]:
    torch.manual_seed(seed)
    state, actions, mask, target = tensors(train, device)
    w = torch.as_tensor(weights(train), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # A warm start must be allowed to keep epoch 0 when every update overfits.
    best = evaluate(model, dev, device)['all']['log_lik']
    best_epoch, bad = 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    g = torch.Generator(device='cpu').manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(target), generator=g).to(device)
        total = 0.
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            logits = model(state[idx], actions[idx], mask[idx])
            loss = (nn.functional.cross_entropy(logits, target[idx], reduction='none') * w[idx]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            total += float(loss.detach()) * len(idx)
        score = evaluate(model, dev, device)['all']['log_lik']
        log(f'  epoch {epoch + 1}: train loss {total / len(target):.4f} dev log-lik {score:.4f}')
        if best is None or score > best + 1e-5:
            best, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {'best_dev_log_lik': best, 'best_epoch': best_epoch}, best_epoch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', choices=['opp', 'ours'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pretrain', nargs='*', default=[])
    ap.add_argument('--init-run', type=Path, help='fine-tune matching seed checkpoints from an existing policy run')
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--dev', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--hidden', type=int, default=192)
    ap.add_argument('--pretrain-epochs', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--pretrain-lr', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch', type=int, default=512)
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

    def concat(paths):
        parts = [load(Path(p)) for p in paths]
        keys = [k for k in parts[0] if k != 'meta']
        data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        data['meta'] = [p['meta'] for p in parts]
        return data
    train, dev = concat(args.train), concat(args.dev)
    pretrain = concat(args.pretrain) if args.pretrain else None
    contracts = {m['feature_schema_sha'] for d in (train, dev, pretrain) if d for m in d['meta']}
    if len(contracts) != 1:
        raise ValueError('training and development tensors use different feature contracts')
    if set(train['group']) & set(dev['group']):
        raise ValueError('a battle appears in both training and development data')
    device = torch.device(args.device)
    state_dim, action_dim = train['state'].shape[1], train['actions'].shape[2]
    initial = None
    if args.init_run:
        initial = json.loads((args.init_run / 'report.json').read_text())
        if (initial['head'] != args.head or
                initial['provenance']['feature_contract'] != next(iter(contracts)) or
                initial['provenance']['dex_sha'] != train['meta'][0]['dex_sha']):
            raise ValueError('initial policy uses a different head or feature contract')
    report = {'schema_version': 1, 'head': args.head, 'provenance': {
        'code': git_commit(), 'feature_contract': train['meta'][0]['feature_schema_sha'],
        'feature_version': train['meta'][0]['feature_version'],
        'js_feature_sha': train['meta'][0]['js_feature_sha'], 'dex_sha': train['meta'][0]['dex_sha'],
        'train': [{'path': p, 'sha': file_hash(Path(p))} for p in args.train],
        'dev': [{'path': p, 'sha': file_hash(Path(p))} for p in args.dev],
        'pretrain': [{'path': p, 'sha': file_hash(Path(p))} for p in args.pretrain],
        'settings': {**vars(args), 'init_run': str(args.init_run) if args.init_run else None},
        'initialization': {'run': str(args.init_run), 'report_sha': file_hash(args.init_run / 'report.json'),
                           'checkpoints': {}} if args.init_run else None,
        'torch': str(torch.__version__), 'device': str(device)},
        'rows': {'train': int(len(train['target'])), 'dev': int(len(dev['target'])),
                 'pretrain': int(len(pretrain['target'])) if pretrain else 0},
        'battles': {'train': int(len(set(train['group']))), 'dev': int(len(set(dev['group'])))},
        'baselines': {}, 'seeds': {}}
    # Baselines on the same development rows: uniform over legal actions.
    legal = dev['mask'].sum(axis=1)
    w = weights(dev, rating=False)
    report['baselines']['uniform'] = {'log_lik': float((np.log(1 / legal) * w / w.sum()).sum()),
                                      'top1': float(((1 / legal) * w / w.sum()).sum())}
    started = time.perf_counter()
    for seed in args.seeds:
        log(f'seed {seed}')
        torch.manual_seed(seed)
        model = PolicyNet(state_dim, action_dim, hidden=args.hidden).to(device)
        stages = {}
        if initial:
            item = initial['seeds'][str(seed)]
            path = args.init_run / item['checkpoint']
            if path.parent != args.init_run or file_hash(path) != item['sha']:
                raise ValueError('modified initial policy checkpoint')
            model, _ = load_checkpoint(path)
            if model.config['state_dim'] != state_dim or model.config['action_dim'] != action_dim:
                raise ValueError('initial policy feature dimensions differ')
            model = model.to(device)
            report['provenance']['initialization']['checkpoints'][str(seed)] = item['sha']
            stages['initial'] = {'dev': evaluate(model, dev, device)}
        if pretrain is not None:
            log(' pretrain')
            stages['pretrain'], _ = fit(model, pretrain, dev, device, args.pretrain_epochs, args.pretrain_lr,
                                        args.patience, args.batch, seed, log)
            stages['pretrain']['dev'] = evaluate(model, dev, device)
        log(' fine-tune')
        stages['finetune'], _ = fit(model, train, dev, device, args.epochs, args.lr, args.patience, args.batch,
                                    seed, log)
        metrics = evaluate(model, dev, device)
        stages['finetune']['dev'] = metrics
        path = out / f'{args.head}-seed{seed}.pt'
        save_checkpoint(path, model, {'head': args.head, 'seed': seed, 'feature_contract': report['provenance'],
                                      'stages': stages})
        export_json(out / f'{args.head}-seed{seed}.json', model,
                    {'head': args.head, 'seed': seed, 'feature_version': report['provenance']['feature_version'],
                     'feature_contract': report['provenance']['feature_contract'], 'checkpoint_sha': file_hash(path)})
        report['seeds'][str(seed)] = {'checkpoint': path.name, 'sha': file_hash(path),
                                      'export': path.with_suffix('.json').name,
                                      'export_sha': file_hash(path.with_suffix('.json')), 'stages': stages}
        log(f" seed {seed}: dev top-1 {metrics['all']['top1']:.3f} top-3 {metrics['all']['top3']:.3f} "
            f"log-lik {metrics['all']['log_lik']:.4f}")
    report['seconds'] = time.perf_counter() - started
    report['dev_mean'] = {k: float(np.mean([s['stages']['finetune']['dev']['all'][k]
                                            for s in report['seeds'].values()])) for k in ('top1', 'top3', 'log_lik')}
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    log(json.dumps({'dev_mean': report['dev_mean'], 'baselines': report['baselines']}))
    log_file.close()


if __name__ == '__main__':
    main()
