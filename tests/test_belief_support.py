import json
import random
from unittest.mock import patch

import numpy as np
import pytest
from test_expert_continuation import GHOLDENGO

from pokesim.belief.sets import SetBelief, Worker, speed_proposals

FAST = {'species': 'Floette-Eternal', 'item': 'Floettite', 'ability': 'Flower Veil', 'nature': 'Timid',
        'moves': ['calmmind', 'moonblast', 'synthesis', 'protect'], 'level': 50,
        'evs': {'hp': 24, 'atk': 0, 'def': 0, 'spa': 10, 'spd': 0, 'spe': 32}}
LOG = ['|poke|p1|Gholdengo, L50|', '|poke|p2|Floette-Eternal, L50|', '|teampreview|3',
       '|switch|p1a: Gholdengo|Gholdengo, L50|175/175',
       '|switch|p2a: Floette|Floette-Eternal, L50|100/100', '|turn|1',
       '|move|p1a: Gholdengo|Nasty Plot|p1a: Gholdengo',
       '|move|p2a: Floette|Calm Mind|p2a: Floette', '|turn|2',
       '|move|p1a: Gholdengo|Flash Cannon|p2a: Floette',
       '|move|p2a: Floette|Moonblast|p1a: Gholdengo', '|turn|3']


class FastSampler:
    def __init__(self):
        self.rng = random.Random(0)

    @staticmethod
    def base_id(species):
        return species

    def sample(self, *args):
        return json.loads(json.dumps(FAST))


@pytest.fixture(scope='module')
def worker():
    worker = Worker()
    yield worker
    worker.close()


def test_all_fast_prior_is_repaired_after_our_repeated_first_move(worker):
    belief = SetBelief([GHOLDENGO], FastSampler(), worker, n=4)
    belief.feed(LOG)
    prior = belief.candidates('floetteeternal', ('calmmind', 'moonblast'), 'Floettite', 'Flower Veil')
    speed = [o for o in belief.observations if o.kind == 'speed']
    assert not belief.speed_compatible(speed, prior).any()
    args = ('floetteeternal', ('calmmind', 'moonblast'), 'Floettite', 'Flower Veil')
    cands = belief.supported_candidates(*args)
    assert len(cands) > len(prior)
    status = belief.speed_support(*args)
    assert status['added'] == 4 and not status['unresolved']
    weights = belief.weights('floetteeternal', cands)
    # No probability floor can revive the fast candidates once coherent slower sets have been generated.
    assert np.all(weights[:4] == 0)
    assert np.isclose(weights.sum(), 1)
    for c in cands[4:]:
        assert c['item'] == 'Floettite' and c['ability'] == FAST['ability'] and c['moves'] == FAST['moves']
        assert max(c['evs'].values()) <= 32 and sum(c['evs'].values()) <= 66
    results = belief.results(speed[-1], cands)
    assert all(r['speeds'][1] <= r['speeds'][0] for r, w in zip(results, weights) if w > 0)
    # The repaired pool and likelihoods are reused for every sampled search world.
    with patch.object(worker, 'run', wraps=worker.run) as run:
        for _ in range(12):
            assert belief.draw(*args) in cands[4:]
        run.assert_not_called()
    # Offline prior/cutoff scoring remains available without conditioning on later observations.
    assert belief.candidates(*args) == prior
    assert np.allclose(belief.weights('floetteeternal', prior, before=0), .25)


def test_impossible_order_history_is_reported_instead_of_claiming_repaired_support(worker):
    belief = SetBelief([GHOLDENGO], FastSampler(), worker, n=2)
    log = LOG[:9] + [LOG[10], LOG[9], LOG[11]]  # reverse the second turn's order, no speed change
    belief.feed(log)
    args = ('floetteeternal', ('calmmind', 'moonblast'), 'Floettite', 'Flower Veil')
    cands = belief.supported_candidates(*args)
    # Exact speed ties can explain either order; rule those out with a tiny known speed change on our side.
    speed = [o for o in belief.observations if o.kind == 'speed']
    assert len(speed) == 2
    # Make the first bound strictly slower than the second in a way no opponent speed satisfies.
    speed[0].sample['p1']['mons'][0]['boosts'] = {'spe': -6}
    belief.cache.clear()
    belief.support_cache.clear()
    cands = belief.supported_candidates(*args)
    assert belief.speed_support(*args)['unresolved']
    assert np.isclose(belief.weights('floetteeternal', cands).sum(), 1)


def test_support_uses_compatible_existing_sets_without_expansion(worker):
    belief = SetBelief([GHOLDENGO], FastSampler(), worker, n=2)
    # Here the original fast set explains moving first.
    belief.feed(LOG[:6] + [LOG[7], LOG[6], LOG[8]])
    args = ('floetteeternal', ('calmmind',), 'Floettite', 'Flower Veil')
    assert len(belief.supported_candidates(*args)) == 2
    assert belief.speed_support(*args)['added'] == 0


def test_speed_proposals_keep_known_items_and_do_not_give_a_mega_a_scarf():
    known = speed_proposals([FAST], unknown_item=False, mega=False)
    mega = speed_proposals([FAST], unknown_item=True, mega=True)
    assert {c['item'] for c in known + mega} == {'Floettite'}
    assert {c['evs']['spe'] for c in known} == set(range(33))
    unknown = speed_proposals([FAST], unknown_item=True, mega=False)
    assert {'', 'Choice Scarf'} <= {c['item'] for c in unknown}
