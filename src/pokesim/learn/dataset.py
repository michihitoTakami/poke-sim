"""Policy training tensors from view rows (pokesim.learn.records / replays): features, legal masks, targets.

  uv run python -m pokesim.learn.dataset --rows data/processed/learn/ladder-train-v1.jsonl.gz ... \
      --out data/processed/learn/features-v1/ladder-train.npz [--sides p2] [--jobs 8]

Each row becomes one example: the state vector, up to MAX_ACTIONS action vectors with a legal mask, the index of
the action taken, the actor's rating, the opponent name, the battle group, the outcome and (ladder p1 rows) the
search teacher's objective per legal action. Rows whose action is not legal in their own view are dropped and
counted. Feature code and schema are hashed into the sidecar so stale tensors are refused.
"""
import argparse
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pokesim.config import ROOT
from pokesim.data.dataset import write_metadata
from pokesim.learn.features import DEX, SCHEMA, VERSION, Features
from pokesim.runtime.bundle import file_hash

MAX_ACTIONS = 12
_FEATURES: Features | None = None


def choices_of(view: dict, legal: list[str]) -> list[tuple[str, str]]:
    """(choice, label) pairs as the worker's legalChoices gives them, from the labels the row kept."""
    req = view['request']
    out = []
    active = (req.get('active') or [{}])[0]
    if not req.get('forceSwitch'):
        for i, m in enumerate(active.get('moves') or []):
            if m.get('disabled'):
                continue
            out.append((f'move {i + 1}', f'move:{m["id"]}'))
            if active.get('canMegaEvo'):
                out.append((f'move {i + 1} mega', f'move:{m["id"]}:mega'))
    if req.get('forceSwitch') or not active.get('trapped'):
        for i, m in enumerate(req['side']['pokemon']):
            if not m.get('active') and not str(m.get('condition', '')).startswith('0'):
                out.append((f'switch {i + 1}', f'switch:{m["details"].split(",")[0].lower()}'))
    # labels in the row's `legal` are authoritative (switch ids are species ids); match by order of kind
    by_label = {label: choice for choice, label in out}
    pairs = []
    for label in legal:
        choice = by_label.get(label)
        if choice is None and label.startswith('switch:'):
            # species ids differ from lower-cased names ("Floette-Eternal" vs floetteeternal): match by position
            species = label.split(':')[1]
            choice = next((c for c, lab in out if lab.startswith('switch:') and
                           lab.split(':')[1].replace('-', '').replace(' ', '') == species), None)
        if choice is None:
            return []
        pairs.append((choice, label))
    return pairs


def encode_row(row: dict) -> dict | None:
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = Features.load()
    pairs = choices_of(row['view'], row['legal'])
    if not pairs or len(pairs) > MAX_ACTIONS or row['action'] not in row['legal']:
        return None
    state, actions = _FEATURES.vectors(row['view'], pairs)
    padded = np.zeros((MAX_ACTIONS, actions.shape[1]), np.float32)
    padded[:len(pairs)] = actions
    mask = np.zeros(MAX_ACTIONS, bool)
    mask[:len(pairs)] = True
    teacher = np.full(MAX_ACTIONS, np.nan, np.float32)
    t = row.get('teacher')
    if t and not t.get('fallback') and not t.get('failed'):
        rho = t.get('rho', .3)
        objective = {e['label']: (1 - rho) * e['expected'] + rho * e['worst'] for e in t['root_evidence']}
        for k, label in enumerate(row['legal']):
            if label in objective:
                teacher[k] = objective[label]
    side = row['side']
    rating = (row.get('rating') or {}).get(side if row.get('source') == 'ladder' else row.get('actor', side))
    return {'state': state, 'actions': padded, 'mask': mask, 'target': row['legal'].index(row['action']),
            'teacher': teacher, 'outcome': float(row['outcome']), 'rating': float(rating or 0),
            'turn': row['turn'], 'group': row['battle_id'].split(':')[1] if ':' in row['battle_id'] else
            row['battle_id'], 'opponent': str(row.get('opponent') or ''), 'side': side,
            'source': row.get('source', ''), 'labels': json.dumps(row['legal'])}


def read_rows(paths: list[Path], sides: set[str] | None):
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if sides and row['side'] not in sides:
                    continue
                yield row


def build(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise ValueError('choose a new output file')
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in args.rows]
    sides = set(args.sides) if args.sides else None
    started = time.perf_counter()
    rows = list(read_rows(paths, sides))
    encoded, dropped = [], 0
    with ProcessPoolExecutor(args.jobs) as ex:
        for item in ex.map(encode_row, rows, chunksize=64):
            if item is None:
                dropped += 1
            else:
                encoded.append(item)
    if not encoded:
        raise ValueError('no usable rows')
    arrays = {k: np.stack([e[k] for e in encoded]) for k in ('state', 'actions', 'mask', 'teacher')}
    for k in ('target', 'outcome', 'rating', 'turn'):
        arrays[k] = np.array([e[k] for e in encoded])
    for k in ('group', 'opponent', 'side', 'source', 'labels'):
        arrays[k] = np.array([e[k] for e in encoded])
    np.savez_compressed(out, **arrays)
    write_metadata(out, {'schema_version': 2, 'feature_version': VERSION, 'feature_schema_sha': file_hash(SCHEMA),
                         'feature_code_sha': file_hash(ROOT / 'src/pokesim/learn/features.py'),
                         'js_feature_sha': file_hash(ROOT / 'services/sim-tools/policy_features.js'),
                         'dex_sha': file_hash(DEX), 'rows': [str(p) for p in paths],
                         'row_shas': {str(p): file_hash(p) for p in paths}, 'sides': sorted(sides or []),
                         'examples': len(encoded), 'dropped': dropped, 'max_actions': MAX_ACTIONS,
                         'seconds': time.perf_counter() - started})
    print(f'{len(encoded)} examples ({dropped} dropped) -> {out} in {time.perf_counter() - started:.0f}s')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sides', nargs='*', default=None, help='keep only these sides (p1: the actor of replay rows)')
    ap.add_argument('--jobs', type=int, default=8)
    build(ap.parse_args())


if __name__ == '__main__':
    main()
