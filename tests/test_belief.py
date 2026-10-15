import json
import random

import numpy as np
import pytest

from pokesim.belief.observe import Observation, Observer
from pokesim.belief.sets import SetBelief, Worker, likelihood
from pokesim.config import ROOT

SETS = json.loads((ROOT / "data/my_teams/baton.json").read_text())
LOG = """|player|p1|me|
|player|p2|opp|
|poke|p1|Skeledirge, L50, F|
|poke|p2|Gliscor, L50, F|
|poke|p2|Pelipper, L50, F|
|poke|p2|Garchomp, L50, F|
|teampreview|3
|start
|switch|p2a: Gliscor|Gliscor, L50, F|100/100
|switch|p1a: Skeledirge|Skeledirge, L50, F|211/211
|turn|1
|move|p2a: Gliscor|Earthquake|p1a: Skeledirge
|-supereffective|p1a: Skeledirge
|-damage|p1a: Skeledirge|73/211
|move|p1a: Skeledirge|Torch Song|p2a: Gliscor
|-crit|p2a: Gliscor
|-damage|p2a: Gliscor|40/100
|-boost|p1a: Skeledirge|spa|1
|
|upkeep
|turn|2
|move|p2a: Gliscor|Swords Dance|p2a: Gliscor
|-boost|p2a: Gliscor|atk|2
|move|p1a: Skeledirge|Shadow Ball|p2a: Gliscor
|-activate|p2a: Gliscor|item: Focus Sash
|-damage|p2a: Gliscor|1/100
|
|upkeep
|error|[Invalid choice] There's nothing to choose
|turn|3""".splitlines()


def base(species: str) -> str:
    return species


def observations(log=LOG):
    obs = Observer(SETS, base)
    obs.feed(log)
    return obs.observations


def test_observer_reads_damage_speed_and_choice_evidence():
    got = observations()
    kinds = [(o.kind, o.turn) for o in got]
    # turn 1: both damage events and the order (it moved first); turn 2: Swords Dance after Earthquake rules out
    # a Choice item, Shadow Ball's damage is not used (Focus Sash), nor is the order (its move was a status move)
    assert kinds == [("damage", 1), ("damage", 1), ("speed", 1), ("nochoice", 2)]
    taken, dealt, speed = got[0], got[1], got[2]
    assert (taken.attacker, taken.move, taken.before, taken.after, taken.crit) == ("p2", "earthquake", 211, 73, False)
    assert (dealt.attacker, dealt.move, dealt.before, dealt.after, dealt.crit) == ("p1", "torchsong", 100, 40, True)
    assert speed.opponent_first is True and speed.species == "gliscor"
    assert taken.sample["p1"]["team"][0]["species"] == "Skeledirge"
    assert taken.sample["p2"]["mons"][0]["hp"] == 1.0
    assert "earthquake" in got[3].revealed[0]


def test_feeding_in_pieces_gives_the_same_observations():
    obs = Observer(SETS, base)
    for k in range(1, len(LOG) + 1):
        obs.feed(LOG[:k])
    assert [(o.kind, o.turn) for o in obs.observations] == [(o.kind, o.turn) for o in observations()]


def test_likelihoods():
    taken = Observation("1:1", "damage", 1, "x", {}, move="m", attacker="p2", before=211, after=73)
    assert likelihood(taken, {"rolls": [140, 139, 138, 138] + [100] * 12, "maxhp": 211}) == 2 / 16
    ko = Observation("1:1", "damage", 1, "x", {}, move="m", attacker="p2", before=125, after=0)
    assert likelihood(ko, {"rolls": [130] * 4 + [120] * 12, "maxhp": 211}) == 4 / 16
    dealt = Observation("1:2", "damage", 1, "x", {}, move="m", attacker="p1", before=100, after=62)
    assert likelihood(dealt, {"rolls": [76] * 8 + [60] * 8, "maxhp": 200}) == 0.5  # 38% lost: 76 of 200
    assert likelihood(dealt, {"rolls": None, "maxhp": 200}) == 0.0  # it could not have been damaged
    assert likelihood(dealt, None) == 1.0  # position not buildable: uninformative
    speed = Observation("1:3", "speed", 1, "x", {}, opponent_first=True)
    assert likelihood(speed, {"speeds": [100, 120]}) == 1.0
    assert likelihood(speed, {"speeds": [120, 100]}) == 0.0
    assert likelihood(speed, {"speeds": [100, 100]}) == 0.5
    nochoice = Observation("2:4", "nochoice", 2, "x", {})
    assert likelihood(nochoice, None, {"item": "Choice Scarf"}) == 0.0
    assert likelihood(nochoice, None, {"item": "Leftovers"}) == 1.0


GLISCOR = {"species": "Gliscor", "ability": "Hyper Cutter", "moves": ["earthquake", "swordsdance", "protect", "toxic"],
           "level": 50}


def gliscor(nature, atk, item="Leftovers"):
    return dict(GLISCOR, nature=nature, item=item, evs={"hp": 32, "atk": atk, "def": 32 - atk, "spa": 0, "spd": 2,
                                                         "spe": 0})


@pytest.fixture(scope="module")
def worker():
    w = Worker()
    yield w
    w.close()


def test_worker_rolls_and_unaware(worker):
    taken = observations()[0]
    sample = json.loads(json.dumps(taken.sample))
    plain = worker.run([dict(taken.job(), candidates=[gliscor("adamant", 32)])])[0][0]
    rolls = plain["rolls"]
    assert len(rolls) == 16 and rolls == sorted(rolls, reverse=True) and plain["maxhp"] == 211
    # Skeledirge's Unaware ignores the attacker's boosts: +2 must not change the damage
    sample["p2"]["mons"][0]["boosts"] = {"atk": 2}
    boosted = worker.run([{"sample": sample, "candidates": [gliscor("adamant", 32)], "attacker": "p2",
                           "move": "earthquake", "crit": False}])[0][0]
    assert boosted["rolls"] == rolls


class FixedSampler:
    """Two candidate sets, alternating: an attacking and a defensive Gliscor."""

    def __init__(self):
        self.rng = random.Random(0)
        self.k = 0

    def base_id(self, species):
        return species

    def sample(self, species, moves=(), item=None, ability=None):
        self.k += 1
        return gliscor("adamant", 32) if self.k % 2 else gliscor("impish", 0)


def test_posterior_prefers_the_set_that_explains_the_damage(worker):
    belief = SetBelief(SETS, FixedSampler(), worker, n=2)
    belief.feed(LOG)
    taken = belief.observations[0]
    cands = belief.candidates("gliscor")
    res = worker.run([dict(taken.job(), candidates=cands)])[0]
    attacking = next(i for i, c in enumerate(cands) if c["nature"] == "adamant")
    roll = res[attacking]["rolls"][8]
    assert roll not in res[1 - attacking]["rolls"]  # the defensive set cannot do this much
    taken.after = taken.before - roll  # the observed damage: one of the attacking set's rolls
    w = belief.weights("gliscor", cands, before=1)
    assert w[attacking] > 0.9 and np.isclose(w.sum(), 1.0)
    assert np.allclose(belief.weights("gliscor", cands, before=0), 0.5)  # no observation yet: the prior
