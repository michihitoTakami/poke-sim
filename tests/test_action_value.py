import json
import subprocess

import numpy as np
import pytest
import torch

from pokesim.config import ROOT
from pokesim.evaluation.branch_matches import digest
from pokesim.evaluation.selection_decisions import paired_gain, read_result, save_result
from pokesim.learn.action_value import guarded_index, load_rows, packed, terminal_loss
from pokesim.learn.branches import CONTINUATIONS


@pytest.mark.parametrize('p,expected', [
    ([[.4, .8], [.4, .9], [.4, .7]], 1),
    ([[.4, .8], [.4, .1], [.4, .7]], 0),
    ([[.4, .44], [.4, .44], [.4, .44]], 0),
])
def test_public_q_ensemble_guard_agrees_in_python_and_javascript(p, expected):
    assert guarded_index(p, 0) == expected
    js = "const {guardedIndex}=require('./services/sim-tools/action_value');" \
         "console.log(guardedIndex(JSON.parse(process.argv[1]),0));"
    result = subprocess.run(['node', '-e', js, json.dumps(p)], cwd=ROOT, check=True,
                            capture_output=True, text=True, timeout=5)
    assert int(result.stdout) == expected


def test_training_balances_source_groups_instead_of_turn_or_branch_counts():
    row = {'state': np.zeros(2, np.float32), 'features': np.zeros((2, 3), np.float32),
           'actions': ['move:a', 'move:b'], 'targets': [1., 0.], 'pair_weights': [[0., 1.], [0., 0.]]}
    batch = packed([dict(row, group=g) for g in ['many', 'many', 'many', 'one']])
    assert batch[-1][:3].sum().item() == pytest.approx(.5)
    assert batch[-1][3].item() == pytest.approx(.5)


def test_advantage_target_removes_common_state_win_potential_and_ignores_padding():
    scores = torch.tensor([[.1, -.1, -1e9]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    pairs = torch.zeros((1, 3, 3))
    first = terminal_loss(scores, torch.tensor([[.7, .5, 0.]]), mask, pairs, advantage=True)
    shifted = terminal_loss(scores, torch.tensor([[.9, .7, 0.]]), mask, pairs, advantage=True)
    assert first.item() == pytest.approx(0., abs=1e-12)
    assert shifted.item() == pytest.approx(first.item(), abs=1e-12)
    first.sum().backward()
    assert scores.grad[0, 2].item() == 0. and torch.isfinite(scores.grad).all()


def test_terminal_interventions_are_teachers_even_when_the_human_chose_the_losing_action(tmp_path, monkeypatch):
    import pokesim.learn.action_value as module

    case = {'id': '1', 'file': 'record.json'}
    cfg = {'continuations': CONTINUATIONS, 'analysis_groups': {'record.json': 'group'},
           'battle_ids': {'record.json': 'battle'}, 'positions': [case]}
    directory = tmp_path / 'train'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({'contract': cfg, 'experiment_id': digest(cfg),
                                                       'information_scope': 'public_observation'}))
    trials = [{'action': a, 'world': w, 'seed': 0, 'continuation': p, 'target': float(a == 'move:a'),
               'trajectory': [{'side': 'p1', 'rootIntervention': True, 'referenceRecommendation': 'move:b',
                               'observation': {'phase': 'move'}}]}
              for a in ['move:a', 'move:b'] for w in range(4) for p in CONTINUATIONS]
    result = {'result': {'actions': ['move:a', 'move:b'], 'trials': trials},
              'simulation_request': {'samples': [{'hypothesis': w} for w in range(4)]},
              'root_provenance': {'executed': 'move:b', 'source': 'human', 'result': 'loss'}}
    monkeypatch.setattr(module, 'load_result', lambda *args: result)
    rows = load_rows(tmp_path, ('train',))['train']
    assert rows[0]['targets'] == [1., 0.]
    assert rows[0]['pair_weights'][0][1] > 0
    trials[0]['target'] = None
    with pytest.raises(ValueError, match='no complete'):
        load_rows(tmp_path, ('train',))


def test_unfinished_factorial_games_stay_bounds_and_missing_pairs_are_rejected():
    before = [{'world': 0, 'target': 1.}, {'world': 1, 'target': 0.}]
    after = [{'world': 0, 'target': None}, {'world': 1, 'target': 1.}]
    result = paired_gain(before, after, ('world',))
    assert result == {'bounds': [0., .5], 'complete': False, 'pairs': 2}
    with pytest.raises(ValueError, match='missing'):
        paired_gain(before, after[:1], ('world',))
    with pytest.raises(ValueError, match='duplicate'):
        paired_gain(before, after + after[:1], ('world',))


def test_audit_resume_rejects_modified_results(tmp_path):
    path = tmp_path / 'case.json'
    save_result(path, {'gain': .1})
    assert read_result(path) == {'gain': .1}
    data = json.loads(path.read_text())
    data['gain'] = .9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='modified'):
        read_result(path)
