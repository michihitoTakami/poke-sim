"""Value training tensors from view rows: the actor's own observation and the real result of that game.

  uv run python -m pokesim.learn.value_dataset --rows data/processed/learn/ladder-train-v1.jsonl.gz \
      --out data/processed/learn/value-v1/ladder-train.npz [--sides p1] [--jobs 4]

Rows are encoded with the frozen value feature encoder (pokesim.value.features, observed-value-v1: public
and private branches with missing-information masks), so a model trained here is loadable by the live
ValueEnsemble. The target is the real outcome for the actor (1 win / 0 loss / .5 draw): no reference policy,
no discount, no material shaping. Weights give every battle a total of 1. Replay rows (both actors of public
games) serve pretraining; ladder rows (us as p1) are the on-distribution set.
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
from pokesim.runtime.bundle import file_hash
from pokesim.value.features import VERSION, Encoder

DEX = ROOT / 'data/processed/public_value/training-dex-v1.json'
_ENCODER: Encoder | None = None


def normalise(view: dict) -> dict:
    """The encoder wants the actor as p1; replay rows seen from p2 are mirrored."""
    if view['side'] == 'p1':
        return view
    swap = {'p1': 'p2', 'p2': 'p1'}
    public = dict(view['public'])
    public['mons'] = {swap[k]: v for k, v in view['public']['mons'].items()}
    public['active'] = {swap[k]: v for k, v in view['public']['active'].items()}
    public['preview'] = {swap[k]: v for k, v in view['public']['preview'].items()}
    public['conditions'] = {swap[k]: v for k, v in (view['public'].get('conditions') or {}).items()}
    request = dict(view['request'])
    request['side'] = dict(request['side'], id='p1')
    return dict(view, side='p1', public=public, request=request)


def encode_row(row: dict) -> dict | None:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = Encoder.load(DEX)
    try:
        public, private, baseline, flags, schema = _ENCODER.encode(normalise(row['view']))
    except ValueError:
        return None
    rating = (row.get('rating') or {}).get(row['side'] if row.get('source') == 'ladder' else row.get('actor', 'p1'))
    return {'public': public, 'private': private, 'baseline': baseline, 'flags': flags,
            'target': float(row['outcome']), 'turn': row['turn'], 'rating': float(rating or 0),
            'group': row['battle_id'], 'source': row.get('source', ''), 'schema': schema}


def build(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise ValueError('choose a new output file')
    out.parent.mkdir(parents=True, exist_ok=True)
    sides = set(args.sides) if args.sides else None
    rows = []
    for path in args.rows:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if (not sides or row['side'] in sides) and row['view'].get('phase') == 'move':
                    rows.append(row)
    started = time.perf_counter()
    encoded, dropped = [], 0
    with ProcessPoolExecutor(args.jobs) as ex:
        for item in ex.map(encode_row, rows, chunksize=64):
            if item is None:
                dropped += 1
            else:
                encoded.append(item)
    if not encoded:
        raise ValueError('no usable rows')
    schema = encoded[0]['schema']
    arrays = {k: np.stack([e[k] for e in encoded]) for k in ('public', 'private', 'baseline', 'flags')}
    for k in ('target', 'turn', 'rating'):
        arrays[k] = np.array([e[k] for e in encoded])
    arrays['group'] = np.array([e['group'] for e in encoded])
    arrays['source'] = np.array([e['source'] for e in encoded])
    groups, counts = np.unique(arrays['group'], return_counts=True)
    per = dict(zip(groups, counts))
    arrays['weight'] = np.array([1 / per[g] for g in arrays['group']], np.float32)
    np.savez_compressed(out, **arrays)
    write_metadata(out, {'schema_version': 2, 'feature_version': VERSION, 'features': schema,
                         'feature_code_sha': file_hash(ROOT / 'src/pokesim/value/features.py'),
                         'dex_sha': file_hash(DEX), 'rows': [str(p) for p in args.rows],
                         'row_shas': {str(p): file_hash(Path(p)) for p in args.rows}, 'sides': sorted(sides or []),
                         'examples': len(encoded), 'battles': int(len(groups)), 'dropped': dropped,
                         'target': 'real outcome of the game for the actor (1 win / 0 loss / .5 draw)',
                         'seconds': time.perf_counter() - started})
    print(f'{len(encoded)} examples from {len(groups)} battles ({dropped} dropped) -> {out} '
          f'in {time.perf_counter() - started:.0f}s')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sides', nargs='*', default=None)
    ap.add_argument('--jobs', type=int, default=4)
    build(ap.parse_args())


if __name__ == '__main__':
    main()
