import numpy as np
import torch

from pokesim.learn.model import PolicyNet
from pokesim.learn.train_policy import fit as fit_policy
from pokesim.learn.train_value import fit as fit_value
from pokesim.value.model import ValueModel


def test_policy_keeps_initial_checkpoint_when_updates_do_not_help():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = PolicyNet(3, 2, hidden=8, embed=4, action_hidden=4, scorer_hidden=4)
    data = {'state': np.ones((2, 3), np.float32), 'actions': np.ones((2, 2, 2), np.float32),
            'mask': np.ones((2, 2), bool), 'target': np.array([0, 1]), 'group': np.array(['a', 'b']),
            'opponent': np.array(['x', 'y']), 'rating': np.array([1100, 1100]), 'turn': np.array([1, 1])}
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    result, epoch = fit_policy(model, data, data, 'cpu', epochs=2, lr=0., patience=1,
                               batch=2, seed=0, log=lambda _: None)
    assert epoch == 0
    assert result['best_epoch'] == 0
    assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())


def test_value_keeps_initial_checkpoint_when_updates_do_not_help():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = ValueModel(3, 2, hidden=8)
    data = {'public': np.ones((2, 3), np.float32), 'private': np.ones((2, 2), np.float32),
            'target': np.array([0., 1.], np.float32), 'weight': np.ones(2, np.float32),
            'group': np.array(['a', 'b'])}
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    result = fit_value(model, data, data, epochs=2, lr=0., patience=1, batch=2, seed=0,
                       device='cpu', log=lambda _: None)
    assert result['best_epoch'] == 0
    assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())
