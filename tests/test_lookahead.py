import json
import subprocess

import numpy as np

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.search.lookahead import solve_zero_sum

EVS_ATK = {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
EVS_BULK = {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 32, "spe": 0}
CHOMP = {"species": "Garchomp", "item": "Life Orb", "ability": "Rough Skin", "nature": "Jolly",
         "moves": ["Swords Dance", "Earthquake", "Dragon Claw", "Protect"], "evs": EVS_ATK, "level": 50}
HIPPO = {"species": "Hippowdon", "item": "Leftovers", "ability": "Sand Stream", "nature": "Impish",
         "moves": ["Earthquake", "Yawn", "Slack Off", "Stealth Rock"], "evs": EVS_BULK, "level": 50}
ROTOM = {"species": "Rotom-Wash", "item": "Sitrus Berry", "ability": "Levitate", "nature": "Bold",
         "moves": ["Hydro Pump", "Volt Switch", "Will-O-Wisp", "Protect"], "evs": EVS_BULK, "level": 50}


def lookahead(sample: dict, seeds: int = 2, rollout: int = 1) -> dict:
    req = {"format": "gen9championsbssregmc", "seeds": seeds, "rollout": rollout, "samples": [sample]}
    out = subprocess.run(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                         input=json.dumps(req) + "\n", capture_output=True, text=True, check=True).stdout
    return json.loads(out)["results"][0]


def test_rock_paper_scissors_is_uniform():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    x, y = solve_zero_sum(rps)
    assert np.allclose(x, 1 / 3, atol=0.03) and np.allclose(y, 1 / 3, atol=0.03)


def test_dominated_row_gets_no_weight():
    x, _ = solve_zero_sum(np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert x[0] > 0.97


def test_finishing_a_low_hp_foe_beats_protect():
    sample = {"p1": {"team": [CHOMP, ROTOM, HIPPO], "mons": [{"hp": 1.0, "boosts": {"atk": 2}}]},
              "p2": {"team": [HIPPO, ROTOM, CHOMP], "mons": [{"hp": 0.1}, {"hp": 0}, {"hp": 0}]}, "field": {}}
    res = lookahead(sample, rollout=0)
    assert res["p2"][-1].startswith("move:")  # no switch options: both bench mons fainted
    value = dict(zip(res["p1"], np.array(res["payoff"]).mean(axis=1)))
    assert value["move:earthquake"] > value["move:protect"] + 2  # the win bonus


def test_levitate_switch_absorbs_earthquake():
    sample = {"p1": {"team": [HIPPO, ROTOM, CHOMP], "mons": [{"hp": 0.3}]},
              "p2": {"team": [CHOMP, ROTOM, HIPPO], "mons": [{"hp": 1.0}]}, "field": {}}
    res = lookahead(sample)
    eq = res["p2"].index("move:earthquake")
    payoff = {lab: row[eq] for lab, row in zip(res["p1"], res["payoff"])}
    assert payoff["switch:rotomwash"] > payoff["move:slackoff"] - 0.5
