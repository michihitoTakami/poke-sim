import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pokesim.evaluation.branch_matches import file_sha
from pokesim.value.dataset import encode_file, inputs, load, prepare, weights_by_battle
from pokesim.value.evaluate import group_means, score
from pokesim.value.features import Encoder, health
from pokesim.value.model import load_checkpoint, predict
from pokesim.value.train import run


def dex():
    return {'format': 'gen9championsbssregmc', 'species': [
        {'id': 'gholdengo', 'name': 'Gholdengo', 'baseSpecies': 'Gholdengo', 'types': ['Steel', 'Ghost'],
         'baseStats': {'hp': 87, 'atk': 60, 'def': 95, 'spa': 133, 'spd': 91, 'spe': 84}}],
        'moves': [{'id': 'recover', 'category': 'Status', 'basePower': 0, 'priority': 0, 'target': 'self',
                   'type': 'Normal'}], 'items': []}


def observation(private=False, hp='100/100'):
    mon = {'species': 'Gholdengo', 'condition': hp, 'boosts': {}, 'volatiles': [], 'moves': []}
    state = {'turn': 2, 'preview': {'p1': ['Gholdengo'], 'p2': ['Gholdengo']},
             'active': {'p1': 'gholdengo', 'p2': 'gholdengo'},
             'mons': {'p1': {'gholdengo': mon}, 'p2': {'gholdengo': dict(mon)}}, 'conditions': {'p1': {}, 'p2': {}}}
    request = None
    if private:
        request = {'side': {'id': 'p1', 'pokemon': [{'details': 'Gholdengo', 'active': True, 'condition': hp,
                                                    'moves': ['recover'], 'item': '', 'ability': 'goodasgold',
                                                    'stats': {'spa': 150}}]},
                   'active': [{'moves': [{'id': 'recover', 'pp': 0, 'maxpp': 8}]}]}
    return {'schema': 1, 'side': 'p1', 'turn': 2, 'phase': 'move', 'public': state, 'request': request}


def test_encoder_masks_missing_private_information_and_zero_pp():
    enc = Encoder(dex())
    a, b = enc.encode(observation()), enc.encode(observation(True))
    assert np.array_equal(a[0], b[0]) and not a[1].any()
    private = dict(zip(b[4]['private'], b[1], strict=True))
    assert private['active.pp_known'] == 1 and private['active.pp_remaining'] == 0
    assert private['active.pp_empty'] == .25 and private['active.item_known'] == 1
    public = dict(zip(a[4]['public'], a[0], strict=True))
    assert public['own.active.item_known'] == 0
    assert health(None) != health('0 fnt')


def test_encoder_uses_only_observation_fields_and_preserves_perspective():
    enc = Encoder(dex())
    obs = observation(True)
    expected = enc.encode(obs)
    altered = copy.deepcopy(obs)
    altered.update(target=0, group='loser', seed=10, root_action='move:recover', source='future')
    altered['request']['opponent'] = {'item': 'future item', 'moves': ['future move']}
    actual = enc.encode(altered)
    assert all(np.array_equal(x, y) for x, y in zip(expected[:4], actual[:4], strict=True))
    altered['request']['side']['id'] = 'p2'
    with pytest.raises(ValueError, match='perspective'):
        enc.encode(altered)


def test_public_pooling_is_order_invariant_and_tracks_revealed_status():
    enc = Encoder(dex())
    obs = observation()
    obs['public']['preview']['p1'].append('Unknown')
    expected = enc.encode(obs)[0]
    obs['public']['preview']['p1'].reverse()
    assert np.array_equal(enc.encode(obs)[0], expected)
    obs['public']['mons']['p2']['gholdengo']['condition'] = '30/100 tox'
    changed = enc.encode(obs)
    assert not np.array_equal(changed[0], expected)
    assert changed[3][3]


def test_battle_weights_and_bootstrap_do_not_treat_turns_as_independent():
    weights = weights_by_battle(['a', 'a', 'b'], [1., 3., 10.])
    assert np.allclose(weights, [.25, .75, 1])
    _, means = group_means([0., 1., 0.], ['a', 'a', 'b'], weights)
    assert np.allclose(means, [.75, 0.])
    data = {'target': np.array([0., 1., 0.]), 'weight': weights, 'group': np.array(['a', 'a', 'b'])}
    result = score({'constant': np.full(3, .5), 'model': np.array([.2, .8, .2])}, data, bootstrap=100)
    assert result['battles'] == 2
    assert result['models']['model']['brier']['mean'] == pytest.approx(.04)
    assert result['paired_differences']['model_minus_constant']['brier']['ci95'][1] < 0


def sources(root):
    base = root / 'values'
    public = base / 'showdown-replays-v2'
    public.mkdir(parents=True)
    pm = {'schema_version': 1, 'information_scope': 'public_state_only',
          'target': 'terminal_outcome_under_replay_players', 'format': 'gen9championsbssregmc',
          'datasets': {}, 'groups': {}, 'rows': {}}
    for j, split in enumerate(('train', 'development')):
        keys = [f'gen9championsbssregmc-{j * 10 + i}' for i in range(1, 3)]
        rows = [{'schema_version': 1, 'split': split, 'group': f'public:{key}', 'target': i,
                 'target_kind': 'terminal_outcome_under_replay_players', 'sample_weight': 1,
                 'observation': observation(hp='80/100' if i else '20/100')}
                for i, key in enumerate(keys)]
        p = public / f'{split}.jsonl.gz'
        with gzip.open(p, 'wt') as f:
            f.write(''.join(json.dumps(row) + '\n' for row in rows))
        pm['datasets'][split] = {'file': p.name, 'sha': file_sha(p)}
        pm['groups'][split], pm['rows'][split] = keys, len(rows)
        rows = []
        for i in range(2):
            group = f'showdown:battle-gen9championsbssregmc-{100 + j * 10 + i}'
            rows.append({'schema_version': 1, 'split': split, 'group': group, 'target': i, 'sample_weight': 1,
                         'step': 1, 'root_action': 'move:recover', 'behavior_action': 'move:recover',
                         'legal_actions': ['move:recover'], 'continuation': 'expert/tactical',
                         'reference_policy_version': 'public-reference-v1',
                         'observation': observation(True, '80/100' if i else '20/100')})
        p = base / f'reference-{split}-v1.jsonl'
        p.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        meta = {'schema_version': 1, 'split': split, 'information_scope': 'public_observation',
                'target': 'terminal_outcome_under_public_reference', 'dataset_sha': file_sha(p),
                'own_reference': 'expert', 'opponent_pool': ['expert/tactical'], 'rows': len(rows),
                'groups': [r['group'] for r in rows],
                'source_scope': {'split': split, 'records': {r['group']: 'sha' for r in rows},
                                 'split_manifest_sha': 'frozen'}}
        Path(str(p) + '.manifest.json').write_text(json.dumps(meta))
    (public / 'manifest.json').write_text(json.dumps(pm))
    return base


def test_source_hashes_and_cross_corpus_overlap_fail_closed(tmp_path):
    base = sources(tmp_path)
    assert len(inputs(base)) == 4
    p = base / 'reference-train-v1.jsonl'
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match='modified'):
        inputs(base)
    base = sources(tmp_path / 'other')
    m = base / 'showdown-replays-v2/manifest.json'
    data = json.loads(m.read_text())
    data['groups']['development'][0] = data['groups']['train'][0]
    m.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='duplicate battle'):
        inputs(base)


def test_prepare_train_reload_and_report_end_to_end(tmp_path):
    base = sources(tmp_path)
    dex_path = tmp_path / 'dex.json'
    dex_path.write_text(json.dumps(dex()))
    cache = tmp_path / 'features'
    meta = prepare(base, dex_path, cache)
    data, loaded = load(cache)
    assert loaded == meta and not data['public_train']['private'].any()
    with pytest.raises(ValueError, match='overwrite'):
        prepare(base, dex_path, cache)
    args = SimpleNamespace(data=cache, out=tmp_path / 'run', device='cpu', threads=1, hidden=8,
                           batch_size=2, public_epochs=2, reference_epochs=2, patience=2,
                           public_lr=.001, reference_lr=.001, bootstrap=20, seeds=[0])
    result = run(args)
    assert result['reference']['battles'] == 2
    assert set(result['reference']['models']) == {'constant', 'hp', 'scratch', 'pretrained', 'transfer'}
    model, checkpoint = load_checkpoint(args.out / 'pretrained-seed0.pt', meta['dataset_id'])
    pred = predict(model, data['reference_development'])
    with np.load(args.out / 'reference-development-predictions.npz') as saved:
        assert np.array_equal(pred, saved['pretrained'])
    assert checkpoint['reference']['version'] == 'public-reference-v1'
    with pytest.raises(ValueError, match='mismatch'):
        load_checkpoint(args.out / 'pretrained-seed0.pt', 'other')
    with pytest.raises(ValueError, match='overwrite'):
        run(args)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())


@pytest.mark.parametrize('setting', ['unknown_outcome', 'root_intervention', 'future_outcome', 'mixed_reference'])
def test_invalid_value_rows_are_rejected_before_training(tmp_path, setting):
    base = sources(tmp_path)
    spec = inputs(base)['reference_train']
    rows = [json.loads(line) for line in spec['path'].read_text().splitlines()]
    if setting == 'unknown_outcome':
        rows[0]['target'] = None
    elif setting == 'root_intervention':
        rows[0]['step'] = 0
    elif setting == 'future_outcome':
        rows[0]['observation']['history'] = ['|turn|2', '|win|Future winner']
    else:
        rows[0]['reference_policy_version'] = 'another-reference'
    spec['path'].write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        encode_file(spec, Encoder(dex()))


def test_modified_feature_cache_is_rejected(tmp_path):
    base = sources(tmp_path)
    dex_path = tmp_path / 'dex.json'
    dex_path.write_text(json.dumps(dex()))
    cache = tmp_path / 'features'
    prepare(base, dex_path, cache)
    (cache / 'reference_train.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='modified prepared'):
        load(cache)


def test_a_real_outcome_value_run_loads_into_the_live_ensemble(tmp_path):
    """A tiny run in the pokesim.learn.train_value layout is accepted and reproduces its torch predictions."""
    import json

    import numpy as np
    import torch
    from test_policy_feature_parity import observe

    from pokesim.config import ROOT
    from pokesim.learn.value_dataset import DEX, encode_row
    from pokesim.runtime.bundle import file_hash
    from pokesim.value.features import Encoder
    from pokesim.value.model import ValueModel
    from pokesim.value.runtime import ValueEnsemble

    req = json.loads((ROOT / 'tests/fixtures/terminal-moonblast-turn14.json').read_text())
    view = observe(req)[0]['p1']['view']
    encoder = Encoder.load(DEX)
    public, private, _, _, features = encoder.encode(view)
    torch.manual_seed(0)
    model = ValueModel(len(public), len(private), 16)
    run = tmp_path / 'value-real-test'
    run.mkdir()
    path = run / 'real-seed0.pt'
    torch.save({'schema_version': 1, 'dataset_id': 'x', 'stage': 'real', 'seed': 0, 'model_config': model.config,
                'state_dict': model.state_dict(), 'features': features, 'reference': {'target': 'real_outcome'}}, path)
    report = {'provenance': {'target': 'real_outcome', 'feature_version': features['version'], 'features': features,
                             'feature_code_sha': file_hash(ROOT / 'src/pokesim/value/features.py'),
                             'dex_sha': file_hash(DEX)},
              'stages': {'real:0': {'checkpoint': 'real-seed0.pt', 'sha': file_hash(path)}}}
    (run / 'report.json').write_text(json.dumps(report))
    ensemble = ValueEnsemble(run)
    assert ensemble.version['reference'] == {'target': 'real_outcome'}
    p = ensemble.probabilities([view])
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.as_tensor(public)[None], torch.as_tensor(private)[None])).numpy()
    assert np.allclose(p, expected[0], atol=1e-6)
    assert encode_row({'view': view, 'outcome': 1, 'turn': 14, 'side': 'p1', 'source': 'ladder',
                       'battle_id': 'b', 'rating': {'p1': 1100}})['target'] == 1
