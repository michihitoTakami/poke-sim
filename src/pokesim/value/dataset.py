"""Validate frozen value artifacts and cache features, weighting each original battle equally."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pokesim.config import FORMAT, ROOT
from pokesim.evaluation.branch_matches import atomic_json, digest, file_sha
from pokesim.evaluation.replay_value import battle_key
from pokesim.value.features import Encoder

NAMES = ('public_train', 'public_development', 'reference_train', 'reference_development')


def weights_by_battle(groups, sample_weights):
    totals = Counter()
    for group, weight in zip(groups, sample_weights, strict=True):
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('sample weights must be finite and positive')
        totals[str(group)] += float(weight)
    return np.asarray([w / totals[str(g)] for g, w in zip(groups, sample_weights, strict=True)], np.float32)


def inputs(base):
    base = Path(base)
    public_dir = base / 'showdown-replays-v2'
    public_meta = public_dir / 'manifest.json'
    pm = json.loads(public_meta.read_text())
    if (pm.get('schema_version') != 1 or pm.get('information_scope') != 'public_state_only' or
            pm.get('target') != 'terminal_outcome_under_replay_players' or pm.get('format') != FORMAT):
        raise ValueError('unsupported public replay artifact')
    specs = {}
    for split in ('train', 'development'):
        path = public_dir / pm['datasets'][split]['file']
        if path.parent != public_dir or path.name != f'{split}.jsonl.gz':
            raise ValueError('unexpected public artifact filename')
        specs[f'public_{split}'] = {'path': path, 'sha': pm['datasets'][split]['sha'],
                                  'manifest': public_meta, 'manifest_sha': file_sha(public_meta),
                                  'split': split, 'groups': ['public:' + g for g in pm['groups'][split]],
                                  'rows': pm['rows'][split], 'scope': 'public_state_only'}
        path = base / f'reference-{split}-v1.jsonl'
        meta_path = Path(str(path) + '.manifest.json')
        meta = json.loads(meta_path.read_text())
        if (meta.get('schema_version') != 1 or meta.get('split') != split or
                meta.get('information_scope') != 'public_observation' or
                meta.get('target') != 'terminal_outcome_under_public_reference' or
                meta.get('source_scope', {}).get('split') != split):
            raise ValueError('unsupported reference value artifact')
        if not set(meta['groups']) <= set(meta['source_scope']['records']):
            raise ValueError('reference groups are outside their frozen source split')
        specs[f'reference_{split}'] = {'path': path, 'sha': meta['dataset_sha'], 'manifest': meta_path,
                                     'manifest_sha': file_sha(meta_path), 'split': split, 'groups': meta['groups'],
                                     'rows': meta['rows'], 'scope': 'public_observation',
                                     'reference': {'own': meta['own_reference'],
                                                   'opponent_pool': meta['opponent_pool']},
                                     'split_manifest_sha': meta['source_scope']['split_manifest_sha']}
    a, b = specs['reference_train'], specs['reference_development']
    if a['reference'] != b['reference'] or a['split_manifest_sha'] != b['split_manifest_sha']:
        raise ValueError('reference policy, opponent pool or source split differs across datasets')
    seen = set()
    for name in NAMES:
        spec = specs[name]
        keys = {battle_key(g) for g in spec['groups']}
        if seen & keys or len(keys) != len(spec['groups']):
            raise ValueError('duplicate battle across value datasets')
        if not keys or spec['rows'] <= 0:
            raise ValueError('empty value dataset')
        seen.update(keys)
        if file_sha(spec['path']) != spec['sha']:
            raise ValueError('modified source value dataset')
    return specs


def encode_file(spec, encoder):
    groups, targets, weights, rows = [], [], [], []
    versions = set()
    seen = set(spec['groups'])
    actual = set()
    schema = None
    opener = gzip.open if spec['path'].suffix == '.gz' else open
    with opener(spec['path'], 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get('schema_version') != 1 or row.get('split') != spec['split'] or
                    row.get('group') not in seen or type(row.get('target')) not in (int, float) or
                    row['target'] not in (0, .5, 1)):
                raise ValueError('invalid value row, target or split membership')
            obs = row['observation']
            if spec['scope'] == 'public_state_only':
                if obs.get('request') is not None or row.get('target_kind') != 'terminal_outcome_under_replay_players':
                    raise ValueError('public pretraining row contains private information or wrong target')
            else:
                if (obs['side'] != 'p1' or row['step'] <= 0 or row.get('root_action') is None or
                        row['behavior_action'] not in row['legal_actions'] or obs.get('request') is None):
                    raise ValueError('invalid non-intervened reference decision')
                if row['continuation'] not in spec['reference']['opponent_pool']:
                    raise ValueError('reference continuation differs from declared opponent pool')
                if row['continuation'].split('/')[0] != spec['reference']['own']:
                    raise ValueError('reference own policy mismatch')
                versions.add(row['reference_policy_version'])
            if any(h.startswith('|win|') or h in ('|tie', '|tie|') for h in obs.get('history', [])):
                raise ValueError('terminal result appears in decision observation')
            public, private, baseline, flags, row_schema = encoder.encode(obs)
            if schema is not None and row_schema != schema:
                raise ValueError('feature schema changed within dataset')
            schema = row_schema
            rows.append((public, private, baseline, flags))
            groups.append(row['group'])
            targets.append(row['target'])
            weights.append(row['sample_weight'])
            actual.add(row['group'])
    if len(rows) != spec['rows'] or actual != seen:
        raise ValueError('source row/group counts differ from artifact metadata')
    if spec['scope'] == 'public_observation' and (len(versions) != 1 or not next(iter(versions))):
        raise ValueError('mixed or missing reference policy version')
    arrays = {k: np.stack([r[i] for r in rows]) for i, k in enumerate(('public', 'private', 'baseline', 'flags'))}
    arrays.update(target=np.asarray(targets, np.float32), weight=weights_by_battle(groups, weights),
                  group=np.asarray(groups))
    return arrays, schema, sorted(versions)


def prepare(base: Path, dex: Path, out: Path):
    if out.exists():
        raise ValueError('refusing to overwrite a prepared value dataset')
    specs = inputs(base)
    encoder = Encoder.load(dex)
    if encoder.dex.get('format') != FORMAT:
        raise ValueError('wrong dex format')
    out.mkdir(parents=True)
    schema, version, artifacts = None, None, {}
    for name in NAMES:
        arrays, names, versions = encode_file(specs[name], encoder)
        if schema is not None and names != schema:
            raise ValueError('feature schema differs across source datasets')
        schema = names
        if name.startswith('reference'):
            if version is not None and versions != version:
                raise ValueError('reference policy version differs across datasets')
            version = versions
        target = out / f'{name}.npz'
        temp = out / f'{name}.npz.tmp'
        with temp.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        temp.rename(target)
        artifacts[name] = {'file': target.name, 'sha': file_sha(target), 'rows': len(arrays['target']),
                           'groups': len(set(arrays['group'])), 'source': {k: str(v) if isinstance(v, Path) else v
                                                                        for k, v in specs[name].items()}}
        print(f'{name}: {artifacts[name]["rows"]} rows / {artifacts[name]["groups"]} battles', flush=True)
    # A matching dex snapshot is kept next to the features for later observation inference.
    (out / 'dex.json').write_bytes(Path(dex).read_bytes())
    contract = {'schema_version': 1, 'features': schema,
                'feature_code_sha': file_sha(Path(__file__).with_name('features.py')),
                'dataset_code_sha': file_sha(Path(__file__)), 'dex_sha': file_sha(out / 'dex.json'),
                'reference': dict(specs['reference_train']['reference'], version=version[0]), 'artifacts': artifacts,
                'weighting': 'input trial-normalized weights, renormalized within each original battle'}
    manifest = dict(contract, dataset_id=digest(contract))
    atomic_json(out / 'manifest.json', manifest)
    return manifest


def load(path):
    path = Path(path)
    meta = json.loads((path / 'manifest.json').read_text())
    if (meta.get('schema_version') != 1 or meta.get('dataset_id') != digest({k: v for k, v in meta.items()
                                                                          if k != 'dataset_id'}) or
            meta['feature_code_sha'] != file_sha(Path(__file__).with_name('features.py')) or
            meta['dataset_code_sha'] != file_sha(Path(__file__)) or file_sha(path / 'dex.json') != meta['dex_sha']):
        raise ValueError('prepared feature contract changed; rebuild into a new directory')
    datasets = {}
    seen = set()
    for name in NAMES:
        item = meta['artifacts'][name]
        target = path / item['file']
        if target.parent != path or target.name != f'{name}.npz' or file_sha(target) != item['sha']:
            raise ValueError('modified prepared features')
        with np.load(target, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        actual = {battle_key(g) for g in arrays['group']}
        if seen & actual or len(actual) != item['groups'] or len(arrays['target']) != item['rows']:
            raise ValueError('prepared split overlap or count mismatch')
        seen.update(actual)
        datasets[name] = arrays
    return datasets, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'data/processed/public_value')
    parser.add_argument('--dex', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    meta = prepare(args.base, args.dex, args.out)
    print('prepared dataset', meta['dataset_id'])


if __name__ == '__main__':
    main()
