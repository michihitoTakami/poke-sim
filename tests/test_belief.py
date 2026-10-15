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


def test_consumed_own_item_is_not_restored():
    start = LOG[:LOG.index("|turn|1") + 1]
    log = start + ["|-enditem|p1a: Skeledirge|Sitrus Berry|[eat]", *LOG[len(start):]]
    got = observations(log)
    assert got[0].sample["p1"]["team"][0]["item"] == ""
    assert next(s for s in SETS if s["species"] == "Skeledirge")["item"] == "Sitrus Berry"


def test_known_historical_item_overrides_current_candidate(worker):
    start = LOG.index("|turn|1")
    log = LOG[:start] + ["|-item|p2a: Gliscor|Choice Scarf|[from] ability: Frisk"] + LOG[start:]
    belief = SetBelief(SETS, FixedSampler(), worker, n=2)
    belief.feed(log)
    speed = next(o for o in belief.observations if o.kind == "speed")
    assert speed.sample["p2"]["mons"][0]["item"] == "choicescarf"
    candidate = gliscor("adamant", 32, "")
    historical = belief.results(speed, [candidate])[0]
    plain_job = json.loads(json.dumps(speed.job()))
    del plain_job["sample"]["p2"]["mons"][0]["item"]
    plain = worker.run([dict(plain_job, candidates=[candidate])])[0][0]
    assert historical["speeds"][1] > plain["speeds"][1]
    assert candidate["item"] == ""
    belief.feed(log + ["|-enditem|p2a: Gliscor|Choice Scarf|[from] move: Knock Off"])
    # Previously known items remain reconstructable after removal.
    assert belief.results(speed, [candidate])[0] == historical


def test_unknown_item_history_is_skipped_after_item_change(worker):
    belief = SetBelief(SETS, FixedSampler(), worker, n=2)
    belief.feed(LOG)
    taken = belief.observations[0]
    cands = belief.candidates("gliscor")
    results = belief.results(taken, cands)
    taken.after = taken.before - results[0]["rolls"][8]
    assert belief.weights("gliscor", cands, before=1)[0] > 0.9
    belief.feed(LOG + ["|-enditem|p2a: Gliscor|Choice Scarf|[from] move: Knock Off"])
    assert np.allclose(belief.weights("gliscor", cands), 0.5)
    # Offline scoring must use the epoch at the scored observation, not the final battle state.
    assert belief.weights("gliscor", cands, before=1, item_epoch=taken.item_epoch)[0] > 0.9


def test_item_change_resets_choice_evidence_and_is_incremental():
    log = LOG + ["|-item|p2a: Gliscor|Choice Scarf|[from] move: Trick",
                 "|move|p2a: Gliscor|Shadow Ball|p1a: Skeledirge", "|turn|4"]
    full = Observer(SETS, base)
    full.feed(log)
    incremental = Observer(SETS, base)
    for k in range(1, len(log) + 1):
        incremental.feed(log[:k])
    for observer in (full, incremental):
        assert observer.item_epochs == {"gliscor": 1}
        assert len([o for o in observer.observations if o.kind == "nochoice"]) == 1
    assert full.observations == incremental.observations


@pytest.mark.parametrize("ability,move,terrain,priorities", [
    ("Overgrow", "grassyglide", "grassyterrain", [0, 1]),
    ("Overgrow", "grassyglide", None, [0, 0]),
    ("Gale Wings", "bravebird", None, [0, 1]),
    ("Flame Body", "bravebird", None, [0, 0]),
    ("Triage", "drainingkiss", None, [0, 3]),
])
def test_modified_priority_is_not_speed_evidence(worker, ability, move, terrain, priorities):
    sample = json.loads(json.dumps(observations()[0].sample))
    sample["field"]["terrain"] = terrain
    species = {"Overgrow": "Rillaboom", "Gale Wings": "Talonflame",
               "Flame Body": "Talonflame", "Triage": "Comfey"}[ability]
    candidate = dict(gliscor("adamant", 32), species=species, ability=ability,
                     moves=[move, "protect", "substitute", "swordsdance"])
    obs = Observation("1:speed", "speed", 1, species.lower(), sample,
                      opponent_first=True, moves=("torchsong", move))
    result = worker.run([dict(obs.job(), candidates=[candidate])])[0][0]
    assert result is not None and result["priorities"] == priorities
    # Deliberately use an order contrary to calculated speed. Only equal priorities can refute it.
    obs.opponent_first = result["speeds"][1] < result["speeds"][0]
    assert likelihood(obs, result) == (0.0 if priorities[0] == priorities[1] else 1.0)


def test_grassy_glide_log_carries_moves_to_priority_worker(worker):
    start = LOG.index("|turn|1")
    log = [line.replace("Gliscor", "Rillaboom") for line in LOG[:start]] + [
        "|-fieldstart|move: Grassy Terrain", "|turn|1",
        "|move|p2a: Rillaboom|Grassy Glide|p1a: Skeledirge",
        "|-damage|p1a: Skeledirge|180/211",
        "|move|p1a: Skeledirge|Torch Song|p2a: Rillaboom",
        "|-damage|p2a: Rillaboom|80/100", "|turn|2"]
    speed = next(o for o in observations(log) if o.kind == "speed")
    assert speed.moves == ("torchsong", "grassyglide")
    candidate = dict(gliscor("adamant", 32), species="Rillaboom", ability="Overgrow")
    result = worker.run([dict(speed.job(), candidates=[candidate])])[0][0]
    assert result["priorities"] == [0, 1]
    assert likelihood(speed, result) == 1.0


def test_known_old_damage_still_informs_stats_after_item_change(worker):
    start = LOG.index("|turn|1")
    log = LOG[:start] + ["|-item|p2a: Gliscor|Choice Band|[from] ability: Frisk"] + LOG[start:]
    belief = SetBelief(SETS, FixedSampler(), worker, n=2)
    belief.feed(log)
    taken = belief.observations[0]
    cands = [gliscor("adamant", 32, ""), gliscor("impish", 0, "")]
    results = belief.results(taken, cands)
    roll = results[0]["rolls"][8]
    assert roll not in results[1]["rolls"]
    taken.after = taken.before - roll
    expected = belief.weights("gliscor", cands, before=1)
    assert expected[0] > 0.8  # a single matching roll still favors the attacking spread
    belief.feed(log + ["|-enditem|p2a: Gliscor|Choice Band|[from] move: Knock Off"])
    assert np.allclose(belief.weights("gliscor", cands, before=1), expected)


def test_known_empty_historical_item_overrides_new_scarf(worker):
    sample = json.loads(json.dumps(observations()[0].sample))
    sample["p2"]["mons"][0]["item"] = ""
    obs = Observation("1:speed", "speed", 1, "gliscor", sample, opponent_first=True,
                      moves=("torchsong", "earthquake"))
    with_scarf = worker.run([dict(obs.job(), candidates=[gliscor("adamant", 32, "Choice Scarf")])])[0][0]
    without = worker.run([dict(obs.job(), candidates=[gliscor("adamant", 32, "")])])[0][0]
    assert with_scarf == without


def test_item_change_turn_has_no_speed_observation():
    change_at = LOG.index("|move|p1a: Skeledirge|Torch Song|p2a: Gliscor")
    log = LOG[:change_at] + ["|-enditem|p2a: Gliscor|Choice Scarf|[from] move: Knock Off"] + LOG[change_at:]
    got = observations(log)
    assert not any(o.kind == "speed" for o in got)
    assert [o.item_epoch for o in got if o.kind == "damage"] == [0, 1]


def test_offline_scoring_does_not_use_future_item_epoch(worker):
    from pokesim.belief.evaluate import score_battle

    prefix = score_battle(SetBelief(SETS, FixedSampler(), worker, n=2), LOG)
    full = score_battle(SetBelief(SETS, FixedSampler(), worker, n=2), LOG + [
        "|-enditem|p2a: Gliscor|Choice Scarf|[from] move: Knock Off"])
    assert full == prefix
