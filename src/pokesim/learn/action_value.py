"""Learn conditional terminal returns for every candidate action; human choices are never targets."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pokesim.evaluation.branch_matches import atomic_json, digest, file_sha, load_result
from pokesim.learn.branches import CONTINUATIONS, confident_pairs, encode_ranks, paired_interval
from pokesim.learn.model import PolicyNet, export_json, load_checkpoint, save_checkpoint

VERSION = 'conditional-terminal-action-value-v1'


def load_rows(data, splits=('train', 'development', 'calibration')):
    """One outcome per complete paired intervention; unfinished games never become losing examples."""
    rows, groups = {}, {}
    contracts = []
    for split in splits:
        directory = data / split
        manifest = json.loads((directory / 'manifest.json').read_text())
        cfg = manifest['contract']
        if manifest['experiment_id'] != digest(cfg) or cfg['continuations'] != CONTINUATIONS:
            raise ValueError('modified or unsupported branch contract')
        if manifest.get('information_scope') != 'public_observation' or cfg.get('continuation_net_sha'):
            raise ValueError('action values require one fixed public reference continuation')
        contracts.append(cfg)
        rows[split], groups[split] = [], set()
        for case in cfg['positions']:
            result = load_result(directory / 'positions' / f"{case['id']}.json", manifest, case)
            trials = result['result']['trials']
            if any(t['target'] is None for t in trials):
                continue
            group = cfg.get('analysis_groups', {}).get(case['file'], cfg['battle_ids'][case['file']])
            groups[split].add(group)
            actions = result['result']['actions']
            root = next(d for d in trials[0]['trajectory'] if d['side'] == 'p1' and d['rootIntervention'])
            reference = root.get('referenceRecommendation')
            if reference not in actions:
                raise ValueError('independent reference action missing')
            rows[split].append({'group': group, 'position': case['id'], 'view': root['observation'],
                'actions': actions, 'baseline_action': reference,
                'targets': [float(np.mean([t['target'] for t in trials if t['action'] == a])) for a in actions],
                'pair_weights': confident_pairs(trials, actions, world_keys=[
                    digest(s) for s in result['simulation_request']['samples']])})
        if not rows[split]:
            raise ValueError('no complete terminal comparisons in ' + split)
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b):
        raise ValueError('source groups cross splits')
    if len({tuple(c['continuations']) for c in contracts}) != 1:
        raise ValueError('continuation league changed')
    return rows


def guarded_index(probabilities, baseline, margin=.05, disagreement=1.64):
    """Ensemble disagreement is a conservative heuristic, not a statistical confidence interval."""
    probabilities = np.asarray(probabilities)
    delta = probabilities - probabilities[:, baseline:baseline + 1]
    lower = delta.mean(0) - disagreement * delta.std(0)
    choice = int(np.argmax(lower))
    return choice if lower[choice] > margin else baseline


def packed(rows):
    size = max(len(r['actions']) for r in rows)
    count, dim = len(rows), rows[0]['features'].shape[1]
    actions, target, mask, pairs = (np.zeros((count, size, dim), np.float32),
        np.zeros((count, size), np.float32), np.zeros((count, size), bool),
        np.zeros((count, size, size), np.float32))
    counts = Counter(r['group'] for r in rows)
    for i, row in enumerate(rows):
        n = len(row['actions'])
        actions[i, :n], target[i, :n], mask[i, :n] = row['features'], row['targets'], True
        pairs[i, :n, :n] = row['pair_weights']
    return (torch.tensor(np.stack([r['state'] for r in rows])), torch.tensor(actions), torch.tensor(mask),
            torch.tensor(target), torch.tensor(pairs),
            torch.tensor([1 / counts[r['group']] / len(counts) for r in rows]))


def probabilities(models, batch):
    with torch.no_grad():
        return np.stack([(model(*batch[:3]) if getattr(model, 'score_transform', 'sigmoid') == 'identity'
                          else torch.sigmoid(model(*batch[:3]))).numpy() for model in models])


def terminal_loss(logits, targets, mask, pairs, *, advantage=False):
    """Center the terminal labels so a state's general win potential cannot dominate action learning."""
    if advantage:
        centered = targets - (targets * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        per_action = (logits - centered).square()
    else:
        per_action = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    fit = (per_action * mask).sum(1) / mask.sum(1)
    gaps = logits[:, :, None] - logits[:, None, :]
    positive = pairs > 0
    ranking = ((gaps - pairs).square() * positive if advantage else nn.functional.softplus(-gaps) * pairs)
    return fit + .25 * ranking.sum((1, 2)) / positive.sum((1, 2)).clamp(min=1)


def metrics(rows, probs, margin, disagreement):
    gains, regrets, groups, changed = [], [], [], []
    for i, row in enumerate(rows):
        n = len(row['actions'])
        baseline = row['actions'].index(row['baseline_action'])
        choice = guarded_index(probs[:, i, :n], baseline, margin, disagreement)
        q = np.asarray(row['targets'])
        gains.append(float(q[choice] - q[baseline]))
        regrets.append(float(q.max() - q[choice]))
        groups.append(row['group'])
        changed.append(choice != baseline)
    unique = sorted(set(groups))
    def grouped(values):
        return float(np.mean([np.mean([v for v, g in zip(values, groups) if g == k]) for k in unique]))
    return {'positions': len(rows), 'groups': len(unique), 'gain': grouped(gains), 'regret': grouped(regrets),
            'changed': sum(changed), 'gain_ci90': paired_interval(gains, groups),
            'scope': 'historical sampled-world root comparisons under fixed reference continuation'}


def train(args):
    if args.out.exists():
        raise ValueError('use a new candidate directory')
    torch.set_num_threads(2)
    training_code_sha = file_sha(Path(__file__))
    # Calibration is opened only after every epoch and hyperparameter has been fixed.
    raw = load_rows(args.data, ('train', 'development'))
    encoded = {k: encode_ranks(v) for k, v in raw.items()}
    batches = {k: packed(v) for k, v in encoded.items()}
    models, epochs = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        first = encoded['train'][0]
        model = PolicyNet(len(first['state']), first['features'].shape[1], hidden=64, embed=32,
                          action_hidden=24, scorer_hidden=32)
        model.score_transform = 'identity' if args.target == 'advantage' else 'sigmoid'
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        best, saved, best_epoch = None, None, 0
        for epoch in range(args.epochs):
            state, actions, mask, targets, pairs, weights = batches['train']
            logits = model(state, actions, mask)
            loss = (terminal_loss(logits, targets, mask, pairs, advantage=args.target == 'advantage') * weights).sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            p = probabilities([model], batches['development'])
            measure = metrics(encoded['development'], p, args.margin, args.disagreement)
            fit_error = float(np.mean([np.mean((p[0, i, :len(r['actions'])] -
                (np.asarray(r['targets']) - (np.mean(r['targets']) if args.target == 'advantage' else 0))) ** 2)
                for i, r in enumerate(encoded['development'])]))
            score = (measure['regret'], fit_error)
            if best is None or score < best:
                best, saved, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
            elif epoch + 1 - best_epoch >= args.patience:
                break
        model.load_state_dict(saved)
        model.eval()
        models.append(model)
        epochs.append(best_epoch)
        print('seed', seed, 'epoch', best_epoch, 'development regret/fit', best, flush=True)
    args.out.mkdir(parents=True)
    from pokesim.learn.features import SCHEMA
    from pokesim.learn.features import VERSION as FEATURE_VERSION

    meta = {'head': 'action_value', 'feature_version': FEATURE_VERSION, 'decision_scope': 'move',
            'target': VERSION, 'continuation': CONTINUATIONS, 'feature_contract': file_sha(SCHEMA),
            'score_transform': 'identity' if args.target == 'advantage' else 'sigmoid', 'objective': args.target,
            'source_split_sha': file_sha(args.data / 'split.json'), 'deployment': 'offline-diagnostic-only'}
    for seed, model in zip(args.seeds, models):
        save_checkpoint(args.out / f'q-seed{seed}.pt', model, meta)
        export_json(args.out / f'q-seed{seed}.json', model, meta)
    calibration = load_rows(args.data)['calibration']
    encoded['calibration'] = encode_ranks(calibration)
    batches['calibration'] = packed(encoded['calibration'])
    report = {'version': VERSION, 'target': 'terminal outcomes of all legal action interventions; no human labels',
        'provenance': meta, 'data': str(args.data.resolve()), 'epochs': epochs, 'training_code_sha': training_code_sha,
        'guard': {'margin': args.margin, 'disagreement': args.disagreement,
                  'scope': 'ensemble heuristic, not a confidence interval'},
        'metrics': {k: metrics(v, probabilities(models, batches[k]), args.margin, args.disagreement)
                    for k, v in encoded.items()},
        'models': {f'q-seed{s}.json': file_sha(args.out / f'q-seed{s}.json') for s in args.seeds},
        'gate': {'passed': False,
                 'reason': 'historical comparisons only; fresh full-policy paired evaluation required'}}
    atomic_json(args.out / 'report.json', report)
    print(json.dumps(report['metrics'], indent=2), flush=True)


def load_models(run):
    report = json.loads((run / 'report.json').read_text())
    models = []
    for name, sha in report['models'].items():
        if file_sha(run / name) != sha:
            raise ValueError('action value model modified')
        model, meta = load_checkpoint((run / name).with_suffix('.pt'))
        if meta.get('target') != VERSION:
            raise ValueError('wrong action value target')
        model.score_transform = meta.get('score_transform', 'sigmoid')
        models.append(model)
    return models, report


def main():
    import atexit

    import poke_env.concurrency as concurrency

    atexit.unregister(concurrency.__clear_loop)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--margin', type=float, default=.05)
    ap.add_argument('--disagreement', type=float, default=1.64)
    ap.add_argument('--target', choices=['advantage', 'return'], default='advantage')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
