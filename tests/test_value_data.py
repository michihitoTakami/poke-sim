import copy
import json
from types import SimpleNamespace

import pytest

from pokesim.evaluation.branch_matches import (
    atomic_json,
    digest,
    ensure_manifest,
    file_sha,
    normalize_cases,
    scoped_cases,
    validate_prior_lineage,
)
from pokesim.evaluation.value_data import export_value


def experiment(root, scope='public_observation', split='train'):
    case = normalize_cases([{'file': 'source.json', 'turn': 1}])[0]
    cfg = {'positions': [case], 'worlds': 1, 'seeds': [0], 'continuations': ['expert/tactical'],
           'information_scope': scope, 'split': split, 'source_scope': {'split': split, 'records': {'battle:1': 'sha'}},
           'battle_ids': {'source.json': 'battle:1'}}
    manifest = ensure_manifest(root, cfg, {})

    def decision(step, side='p1', phase='move'):
        obs = {'schema': 1, 'side': side, 'request': {'side': {'id': side}}, 'history': ['|turn|1']}
        return {'step': step, 'side': side, 'phase': phase, 'rootIntervention': step == 0, 'action': 'move:x',
                'legal': ['move:x'], 'target': 1 if side == 'p1' else 0, 'observation': obs}

    trajectory = [decision(0), decision(0, 'p2'), decision(1), decision(1, 'p2'),
                  decision(2, phase='switch'), decision(3)]
    trial = {'world': 0, 'seed': 0, 'continuation': 'expert/tactical', 'action': 'move:x', 'outcome': 'win',
             'target': 1, 'error': None, 'trajectory': trajectory}
    row = {'experiment_id': manifest['experiment_id'], 'case': case,
           'result': {'informationScope': scope, 'policyVersion': 'public-reference-v1',
                      'actions': ['move:x'], 'trials': [trial]}}
    row['payload_sha'] = digest(row)
    atomic_json(root / 'positions' / f"{case['id']}.json", row)
    return manifest, row


def test_value_export_keeps_terminal_observations_but_excludes_forced_root_opponent_and_switch_decisions(tmp_path):
    root = tmp_path / 'experiment'
    experiment(root)
    target = tmp_path / 'train.jsonl'
    meta = export_value(root, target)
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == meta['rows'] == 2
    assert [r['step'] for r in rows] == [1, 3]
    assert all(r['root_action'] == 'move:x' for r in rows)
    assert all(r['observation']['side'] == 'p1' and r['target'] == 1 for r in rows)
    assert all(r['group'] == 'battle:1' and r['split'] == 'train' for r in rows)
    assert sum(r['sample_weight'] for r in rows) == pytest.approx(1)
    assert meta['dataset_sha'] == file_sha(target)
    assert json.loads(target.with_suffix('.jsonl.manifest.json').read_text()) == meta
    with pytest.raises(ValueError, match='overwrite'):
        export_value(root, target)


def test_rejected_action_attempt_is_not_exported_as_an_executed_reference_decision(tmp_path):
    root = tmp_path / 'experiment'
    _, row = experiment(root)
    row['result']['trials'][0]['trajectory'][2]['rejected'] = 'trapped'
    row['payload_sha'] = digest({k: v for k, v in row.items() if k != 'payload_sha'})
    atomic_json(root / 'positions' / f"{row['case']['id']}.json", row)
    target = tmp_path / 'value.jsonl'
    export_value(root, target)
    samples = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(samples) == 1 and samples[0]['step'] == 3 and samples[0]['sample_weight'] == 1


@pytest.mark.parametrize('scope,split', [
    ('sampled_full_information_continuations', 'train'), ('public_observation', 'diagnostic'),
    ('public_observation', 'test'),
])
def test_full_information_diagnostics_and_test_data_cannot_be_exported_as_training(tmp_path, scope, split):
    root = tmp_path / 'experiment'
    experiment(root, scope, split)
    with pytest.raises(ValueError, match='scoped public'):
        export_value(root, tmp_path / 'train.jsonl')
    assert not (tmp_path / 'train.jsonl').exists()


def test_censored_trajectory_never_becomes_training_data(tmp_path):
    root = tmp_path / 'experiment'
    manifest, row = experiment(root)
    trial = row['result']['trials'][0]
    trial.update(target=None, outcome='unfinished')
    for d in trial['trajectory']:
        d['target'] = None
    row['payload_sha'] = digest({k: v for k, v in row.items() if k != 'payload_sha'})
    atomic_json(root / 'positions' / f"{row['case']['id']}.json", row)
    with pytest.raises(ValueError, match='no non-intervened'):
        export_value(root, tmp_path / 'train.jsonl')
    assert not (tmp_path / 'train.jsonl').exists()


def test_a_wrong_perspective_target_or_illegal_action_invalidates_export(tmp_path):
    root = tmp_path / 'experiment'
    _, row = experiment(root)
    row['result']['trials'][0]['trajectory'][1]['target'] = 1
    row['payload_sha'] = digest({k: v for k, v in row.items() if k != 'payload_sha'})
    atomic_json(root / 'positions' / f"{row['case']['id']}.json", row)
    with pytest.raises(ValueError, match='terminal outcome'):
        export_value(root, tmp_path / 'train.jsonl')


def test_source_split_membership_is_validated_before_output(monkeypatch, tmp_path):
    records = tmp_path / 'data/raw/ladder'
    records.mkdir(parents=True)
    source = records / 'train.json'
    source.write_text(json.dumps({'log': ['|turn|1'], 'lineup': ['Garchomp']}))
    primary = {'path': 'data/raw/ladder/train.json', 'sha': file_sha(source)[:16]}
    battle = SimpleNamespace(battle_id='battle:train', source='showdown', records=[primary])

    class Scope:
        def __init__(self, *args):
            pass

        def select(self, split):
            assert split == 'train'
            return [battle]

        def records(self, split):
            return [('battle:train', json.loads(source.read_text()))]

        def metadata(self, split):
            return {'split': split, 'records': {'battle:train': primary['sha']},
                    'records_sha': 'records', 'split_manifest_sha': 'manifest'}

    monkeypatch.setattr('pokesim.data.dataset.DatasetScope', Scope)
    args = SimpleNamespace(split='train', information_scope='public_observation', data_root=tmp_path,
                           records=records, positions=None, ledger=None, split_manifest=None)
    cases, contract = scoped_cases(args)
    assert cases[0]['file'] == 'train.json'
    assert contract['battle_ids'] == {'train.json': 'battle:train'}
    explicit = tmp_path / 'cases.json'
    explicit.write_text(json.dumps([{'file': 'test.json', 'turn': 1}]))
    args.positions = explicit
    with pytest.raises(ValueError, match='outside.*train'):
        scoped_cases(args)
    args.information_scope = 'sampled_full_information_continuations'
    with pytest.raises(ValueError, match='public_observation'):
        scoped_cases(args)


def test_missing_or_mixed_reference_policy_cannot_silently_export(tmp_path):
    root = tmp_path / 'experiment'
    manifest, _ = experiment(root)
    cfg = copy.deepcopy(manifest['contract'])
    cfg['continuations'].append('tactical/disrupt')
    manifest.update(contract=cfg, experiment_id=digest(cfg))
    atomic_json(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='one fixed'):
        export_value(root, tmp_path / 'train.jsonl')


def test_prior_train_membership_must_match_the_source_split_contract():
    contract = {'source_scope': {'split_manifest_sha': 'manifest'}, 'prior_train_records_sha': 'train-records'}
    validate_prior_lineage({'split_manifest_sha': 'manifest', 'records_sha': 'train-records'}, contract)
    with pytest.raises(ValueError, match='different train split'):
        validate_prior_lineage({'split_manifest_sha': 'other', 'records_sha': 'train-records'}, contract)
    with pytest.raises(ValueError, match='different train split'):
        validate_prior_lineage({'split_manifest_sha': 'manifest', 'records_sha': 'other-records'}, contract)


def test_candidate_branches_keep_distinct_trajectory_keys_and_normalized_weights(tmp_path):
    root = tmp_path / 'experiment'
    _, row = experiment(root)
    other = copy.deepcopy(row['result']['trials'][0])
    other['action'] = 'move:y'
    row['result']['actions'].append('move:y')
    row['result']['trials'].append(other)
    row['payload_sha'] = digest({k: v for k, v in row.items() if k != 'payload_sha'})
    atomic_json(root / 'positions' / f"{row['case']['id']}.json", row)
    target = tmp_path / 'train.jsonl'
    export_value(root, target)
    weights = {}
    for r in map(json.loads, target.read_text().splitlines()):
        key = (r['position_id'], r['world'], r['seed'], r['continuation'], r['root_action'])
        weights[key] = weights.get(key, 0) + r['sample_weight']
    assert len(weights) == 2 and all(w == pytest.approx(1) for w in weights.values())
