"""Gate A (offline): the learned opponent policy against the deployed prediction on the same held-out battles.

  uv run python -m pokesim.learn.evaluate_opponent --run runs/opp-policy-v1 \
      --dev data/processed/learn/features-v1/ladder-development-p2.npz [--out runs/opp-policy-v1/gate.json]

The net's top-1 / top-3 / mean log likelihood come from its checkpoints on the development tensors (battle
weighted, by rating band and by phase). The incumbent is what the live advisor recorded at the same battles'
turns (advice[].opponent: the search prediction blended with the cloned PPO policy), scored like
scripts/ladder_report.py (Mega variants merged; probabilities floored at 1e-3 for the log likelihood). The two
row sets differ slightly (the net's rows need the opponent's action to be legal in its sampled world), so the
comparison is indicative; the gate thresholds are absolute.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pokesim.data.curate_battles import CURATED
from pokesim.data.dataset import DatasetScope
from pokesim.learn.model import load_checkpoint
from pokesim.learn.train_policy import evaluate, load

GATE = {'top1': .46, 'top3': .72}


def match(pred: str, actual: str) -> bool:
    p, a = pred.split(':'), actual.split(':')
    return p[0] == a[0] and (p[1] == a[1] or p[1].startswith(a[1]) or a[1].startswith(p[1]))


def opponent_action(log: list[str], turn: int) -> str | None:
    start = next((i for i, ln in enumerate(log) if ln == f'|turn|{turn}'), None)
    if start is None:
        return None
    megas = False
    for line in log[start + 1:]:
        parts = line.split('|')
        if parts[1:2] == ['turn']:
            return None
        if parts[1:2] == ['-mega'] and parts[2].startswith('p2'):
            megas = True
        if line.startswith(('|move|p2', '|switch|p2')) and '[from]' not in line and not line.endswith('[still]'):
            if parts[1] == 'move':
                return 'move:' + parts[3].lower().replace(' ', '').replace('-', '') + (':mega' if megas else '')
            return 'switch:' + parts[3].split(',')[0].lower().replace(' ', '').replace('-', '')
    return None


def incumbent(records: list[tuple[str, dict]]) -> dict:
    """The recorded live prediction's accuracy on these battles' normal turns, battle-weighted."""
    per_battle = []
    for _, rec in records:
        hits = []
        for a in rec.get('advice', []):
            if a.get('force') or not a.get('opponent'):
                continue
            actual = opponent_action(rec['log'], a['turn'])
            if actual is None:
                continue
            preds = a['opponent']
            p = next((q for lab, q in preds if match(lab, actual)), 0.)
            hits.append((match(preds[0][0], actual), any(match(lab, actual) for lab, _ in preds[:3]),
                         math.log(max(1e-3, p))))
        if hits:
            per_battle.append(np.mean(hits, axis=0))
    if not per_battle:
        return {}
    top1, top3, ll = np.mean(per_battle, axis=0)
    return {'battles': len(per_battle), 'top1': float(top1), 'top3': float(top3), 'log_lik': float(ll),
            'note': 'recorded advice[].opponent (search + cloned policy blend), floored at 1e-3'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--dev', nargs='+', required=True)
    ap.add_argument('--ledger', default=str(CURATED / 'v2/battles.jsonl'))
    ap.add_argument('--manifest', default=str(CURATED / 'split_manifest_v2.json'))
    ap.add_argument('--split', default='development')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run = Path(args.run)
    report = json.loads((run / 'report.json').read_text())
    parts = [load(Path(p)) for p in args.dev]
    dev = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != 'meta'}
    device = torch.device('cpu')
    seeds = {}
    for seed, item in report['seeds'].items():
        model, _ = load_checkpoint(run / item['checkpoint'])
        seeds[seed] = evaluate(model, dev, device)
    mean = {k: float(np.mean([s['all'][k] for s in seeds.values()])) for k in ('top1', 'top3', 'log_lik')}
    scope = DatasetScope(ledger=Path(args.ledger), manifest=Path(args.manifest))
    records = [(b, rec) for b, rec in scope.records(args.split) if (rec.get('source') or '').startswith('showdown')]
    base = incumbent(records)
    out = {'run': str(run), 'head': report['head'], 'dev_rows': int(len(dev['target'])),
           'dev_battles': int(len(set(dev['group']))), 'net': {'mean': mean, 'seeds': seeds},
           'incumbent': base, 'uniform': report['baselines'].get('uniform'),
           'gate': {'thresholds': GATE, 'top1': mean['top1'] >= GATE['top1'], 'top3': mean['top3'] >= GATE['top3'],
                    'beats_incumbent_log_lik': bool(base) and mean['log_lik'] >= base.get('log_lik', 0) + .10,
                    'by_rating': {k: (v['top1'] if v else None) for k, v in next(iter(seeds.values())).items()
                                  if k.startswith('rating')}}}
    out['gate']['passed'] = out['gate']['top1'] and out['gate']['top3']
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == '__main__':
    main()
