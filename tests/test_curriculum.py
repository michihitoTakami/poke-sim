import copy
from types import SimpleNamespace

import pytest

from pokesim.evaluation.branch_matches import restore_private_request
from pokesim.learn.curriculum import components, decisions


def test_opponent_and_preview_groups_are_transitive_and_order_independent():
    records = {'a': {'opponent': 'One', 'preview': {'species': ['x', 'y']}},
               'b': {'opponent': 'one', 'preview': {'species': ['z', 'w']}},
               'c': {'opponent': 'Two', 'preview': {'species': ['w', 'z']}},
               'd': {'opponent': 'Three', 'preview': {'species': ['different']}}}
    expected = {frozenset(['a', 'b', 'c']), frozenset(['d'])}
    assert {frozenset(g) for g in components(records)} == expected
    assert {frozenset(g) for g in components(dict(reversed(list(records.items()))))} == expected


def test_exact_private_request_uses_this_turn_and_normalizes_our_wire_side_without_mutation():
    original = {'side': {'id': 'p2', 'pokemon': [{'ident': 'p2: Own', 'condition': '12/150'}]},
                'active': [{'moves': [{'id': 'protect', 'pp': 2}]}]}
    rec = {'advice': [{'turn': 3, 'request': original}, {'turn': 4, 'request': {'future': True}}]}
    saved = copy.deepcopy(rec)
    result = []
    restore_private_request(SimpleNamespace(parse_request=result.append), rec, 3)
    assert result[0]['side']['id'] == 'p1' and result[0]['side']['pokemon'][0]['ident'] == 'p1: Own'
    assert result[0]['active'][0]['moves'][0]['pp'] == 2 and rec == saved
    with pytest.raises(ValueError, match='missing exact'):
        restore_private_request(SimpleNamespace(), rec, 5)


def test_decision_mining_is_not_driven_by_wins_or_manual_recommendation_disagreement():
    private = {'active': [{}], 'side': {'pokemon': [{}, {}, {}]}}
    rec = {'result': 'loss', 'log': ['|turn|1', '|turn|2'], 'advice': [
        {'turn': t, 'request': private, 'chosen': 'move:protect'} for t in [1, 2]]}
    before = decisions(rec)
    rec['result'] = 'win'
    rec['advice'][0]['submission'] = {'source': 'manual', 'action': 'move:other'}
    assert decisions(rec) == before


def test_trajectory_sampling_preserves_terminal_results_and_keeps_only_our_bounded_states():
    from test_branch_matches import request, worker
    from test_expert_continuation import FLOETTE, HIPPO, side

    req = request({'p1': side([FLOETTE]), 'p2': side([HIPPO]), 'field': {}}, maxTurns=12,
        actions=['move:substitute'], continuations=['planned/counterplay'],
        informationScope='public_observation', observationLog=[
            '|switch|p1a: Floette|Floette-Eternal, L50|100/100',
            '|switch|p2a: Hippowdon|Hippowdon, L50|100/100', '|turn|1'])
    full = worker(req)
    req['branches']['trajectorySample'] = {'continuations': 2}
    sampled = worker(req)
    assert 'error' not in sampled
    assert [(r['outcome'], r['target'], r['opponentAction']) for r in sampled['trials']] == [
           (r['outcome'], r['target'], r['opponentAction']) for r in full['trials']]
    for row in sampled['trials']:
        assert 1 <= len(row['trajectory']) <= 3
        assert all(d['side'] == 'p1' for d in row['trajectory'])
        assert row['trajectory'][0]['rootIntervention']


def test_fixed_policy_does_not_become_its_own_reference_teacher(tmp_path):
    import json

    import torch
    from test_branch_matches import request, worker
    from test_expert_continuation import CHOMP, HIPPO, side

    from pokesim.learn.features import SCHEMA, VERSION
    from pokesim.learn.model import PolicyNet, export_json

    schema = json.loads(SCHEMA.read_text())
    model = PolicyNet(len(schema['state']), len(schema['action']), hidden=8, embed=4, action_hidden=4, scorer_hidden=4)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    path = tmp_path / 'fixed.json'
    export_json(path, model, {'feature_version': VERSION})
    req = request({'p1': side([dict(CHOMP, moves=['Swords Dance', 'Dragon Claw'], item='')]),
                   'p2': side([HIPPO], [{'hp': .001}]), 'field': {}},
                  actions=['move:dragonclaw'], continuations=['planned/counterplay'], maxTurns=5,
                  informationScope='public_observation', policyNets={'ours': str(path)}, observationLog=[
                      '|switch|p1a: Garchomp|Garchomp, L50|100/100',
                      '|switch|p2a: Hippowdon|Hippowdon, L50|1/100', '|turn|1'])
    result = worker(req)
    assert 'error' not in result
    root = next(d for d in result['trials'][0]['trajectory'] if d['side'] == 'p1')
    assert root['policyRecommendation'] == 'move:swordsdance'
    assert root['referenceRecommendation'] == 'move:dragonclaw'


def test_training_reads_the_independent_reference_recommendation(tmp_path, monkeypatch):
    import json

    from pokesim.evaluation.branch_matches import digest
    from pokesim.learn import branches

    cfg = {'continuations': branches.CONTINUATIONS, 'positions': [{'file': 'battle.json', 'id': 'one'}],
           'battle_ids': {'battle.json': 'battle'}, 'analysis_groups': {'battle.json': 'opponent-cluster'},
           'continuation_net_sha': 'fixed'}
    directory = tmp_path / 'train'
    (directory / 'positions').mkdir(parents=True)
    (directory / 'manifest.json').write_text(json.dumps({'contract': cfg, 'experiment_id': digest(cfg),
                                                       'information_scope': 'public_observation'}))
    (directory / 'positions/one.json').write_text('{}')
    result = {'result': {'continuationNet': 'fixed', 'policyVersion': 'fixed-policy', 'actions': ['move:protect'],
               'trials': [{'world': 0, 'seed': 0, 'continuation': 'planned/counterplay',
                           'action': 'move:protect', 'target': 1., 'trajectory': [
                               {'side': 'p1', 'rootIntervention': True, 'phase': 'move', 'observation': {},
                                'policyRecommendation': 'move:protect',
                                'referenceRecommendation': 'move:batonpass'}]}]}}
    monkeypatch.setattr(branches, 'load_result', lambda *args: result)
    monkeypatch.setattr(branches, 'development_files', set)
    ranks, _, _ = branches.examples(tmp_path)
    assert ranks['train'][0]['baseline_action'] == 'move:batonpass'
    assert ranks['train'][0]['group'] == 'opponent-cluster'


def test_grouped_value_cannot_load_without_policy_crossplay_even_if_value_gate_was_marked_passed(tmp_path):
    import json

    from pokesim.value.runtime import ValueEnsemble

    (tmp_path / 'report.json').write_text(json.dumps({'provenance': {
        'target': 'terminal_branch_continuation', 'grouped_curriculum_sha': 'grouped',
        'reference': {'policy_sha': 'fixed'}}, 'gate': {'passed': True}}))
    with pytest.raises(ValueError, match='reference continuation'):
        ValueEnsemble(tmp_path)


def test_promotion_counts_independent_groups_and_rejects_missing_or_different_states(tmp_path, monkeypatch):
    import json

    from pokesim.learn import promotion

    old, new = tmp_path / 'old', tmp_path / 'new'
    cases = [{'id': str(i), 'file': f'{i}.json'} for i in range(20)]
    cfg = dict(positions=cases, world_seed=0, worlds=4, seeds=[0, 1], continuations=['planned/counterplay'],
               analysis_groups={c['file']: c['id'] for c in cases}, records={},
               prior_sha='same', worker_code_sha='same')
    for path in [old, new]:
        (path / 'calibration/positions').mkdir(parents=True)
        (path / 'calibration/manifest.json').write_text(json.dumps({'contract': cfg}))
        for case in cases:
            (path / 'calibration/positions' / f"{case['id']}.json").write_text('{}')
    def loaded(path, manifest, case):
        fresh = path.is_relative_to(new)
        return {'simulation_request': {'samples': ['same']}, 'result': {'trials': [
            {'action': 'move:attack' if fresh else 'move:setup', 'world': w, 'seed': s,
             'continuation': 'planned/counterplay', 'target': float(fresh), 'trajectory': [
                 {'side': 'p1', 'rootIntervention': True, 'referenceRecommendation': 'move:setup',
                  'policyRecommendation': 'move:attack' if fresh else 'move:setup'}]}
            for w in range(4) for s in [0, 1]]}}
    monkeypatch.setattr(promotion, 'load_result', loaded)
    assert promotion.compare(old, new)['passed']
    (new / 'calibration/positions/0.json').unlink()
    result = promotion.compare(old, new)
    assert not result['passed'] and result['incomplete_cases'] == ['0']
    (new / 'calibration/positions/0.json').write_text('{}')
    cfg['analysis_groups'] = {c['file']: 'one-opponent-group' for c in cases}
    for path in [old, new]:
        (path / 'calibration/manifest.json').write_text(json.dumps({'contract': cfg}))
    assert not promotion.compare(old, new)['passed']
    def different(path, manifest, case):
        result = loaded(path, manifest, case)
        if path.is_relative_to(new):
            result['simulation_request']['samples'] = ['different']
        return result
    monkeypatch.setattr(promotion, 'load_result', different)
    with pytest.raises(ValueError, match='sampled states differ'):
        promotion.compare(old, new)


def test_move_only_actor_does_not_choose_the_baton_receiver(tmp_path):
    import json

    import torch
    from test_branch_matches import request, worker
    from test_expert_continuation import CHOMP, FLOETTE, HIPPO, PASSER, side

    from pokesim.learn.features import SCHEMA, VERSION
    from pokesim.learn.model import PolicyNet, export_json

    schema = json.loads(SCHEMA.read_text())
    model = PolicyNet(len(schema['state']), len(schema['action']), hidden=8, embed=4, action_hidden=4, scorer_hidden=4)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    path = tmp_path / 'move-only.json'
    export_json(path, model, {'feature_version': VERSION, 'decision_scope': 'move'})
    req = request({'p1': side([PASSER, CHOMP, FLOETTE],
                             [{'hp': 1, 'boosts': {'def': 2, 'spe': 2}}, {'hp': 1}, {'hp': 1}]),
                   'p2': side([dict(HIPPO, moves=['Splash'])]), 'field': {}},
                  actions=['move:batonpass'], continuations=['planned/counterplay'], maxTurns=5,
                  informationScope='public_observation', policyNets={'ours': str(path)}, observationLog=[
                      '|switch|p1a: Scolipede|Scolipede, L50|100/100',
                      '|switch|p2a: Hippowdon|Hippowdon, L50|100/100', '|turn|1'])
    result = worker(req)
    assert 'error' not in result
    assert all(t['outcome'] != 'failed' for t in result['trials'])
    assert any(d['phase'] == 'switch' and d['side'] == 'p1'
               for t in result['trials'] for d in t['trajectory'])


def test_continuation_value_binds_its_actor_and_opponent_rollout_league():
    from pokesim.live.advisor import branch_rollouts

    model = SimpleNamespace(version={'reference': {'policy_sha': 'fixed',
        'opponent_pool': ['planned/tactical', 'planned/counterplay']}})
    settings = branch_rollouts(model)
    assert settings['rolloutPolicies'] == {'p1': ['planned'], 'p2': ['tactical', 'counterplay']}
    assert settings['rolloutMix'] == {'ours': 1., 'opp': 0.}
    model.version['reference']['opponent_pool'] = ['different/expert']
    with pytest.raises(ValueError, match='unsupported'):
        branch_rollouts(model)


def test_terminal_review_uses_the_same_fixed_own_actor_as_its_value_search():
    import time

    from pokesim.search.review import review

    captured = []
    def query(request):
        captured.append(request)
        raise TimeoutError('test stops before simulation')
    engine = SimpleNamespace(_query=query, query_timeout=30.)
    review(engine, {'format': 'format', 'samples': [{}], 'observationLog': [],
                   'policyNets': {'ours': '/fixed/ours.json', 'opp': '/old/opp.json'}},
           [{'p1': ['move:a', 'move:b'], 'terminal_search': {}}], time.monotonic() + 2)
    assert captured[0]['branches']['policyNets'] == {'ours': '/fixed/ours.json'}
