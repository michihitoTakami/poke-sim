import itertools

import numpy as np
import pytest

from pokesim.learn.branches import confident_pairs


def trials(worlds=8):
    return [dict(world=w, seed=s, continuation=p, action=a, target=1. if a == 'attack' else 0.)
            for w, s, p, a in itertools.product(range(worlds), range(4),
                                                ['planned/disrupt', 'planned/counterplay'], ['attack', 'setup'])]


def test_confident_comparisons_keep_winning_setup_and_reject_a_single_lucky_world():
    rows = trials()
    assert confident_pairs(rows, ['attack', 'setup']) == [[0, 1], [0, 0]]
    # It is outcome evidence, not an attack preference. Safe winning setup teaches the reverse ranking.
    for r in rows:
        r['target'] = 1 - r['target']
    assert confident_pairs(rows, ['attack', 'setup']) == [[0, 0], [1, 0]]
    for r in rows:
        r['target'] = float(r['world'] == 0 and r['action'] == 'setup')
    assert not np.asarray(confident_pairs(rows, ['attack', 'setup'])).any()


def test_repeated_chance_seeds_do_not_replace_hidden_world_coverage():
    assert not np.asarray(confident_pairs(trials(1) * 1, ['attack', 'setup'])).any()


def test_censoring_missing_pairs_and_stronger_opponent_counterplay_cannot_teach_false_certainty():
    rows = trials()
    rows[0]['target'] = None
    assert not np.asarray(confident_pairs(rows, ['attack', 'setup'])).any()
    rows = trials()[1:]
    assert not np.asarray(confident_pairs(rows, ['attack', 'setup'])).any()
    rows = trials()
    for r in rows:
        if r['continuation'] == 'planned/counterplay':
            r['target'] = 1 - r['target']
    assert not np.asarray(confident_pairs(rows, ['attack', 'setup'])).any()
    with pytest.raises(ValueError, match='duplicate'):
        confident_pairs(rows + rows[:1], ['attack', 'setup'])
