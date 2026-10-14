"""Continuation policies, terminal scoring and the lost-win-path check of the lookahead worker."""
import json
import subprocess

import numpy as np

from pokesim.config import ROOT, SHOWDOWN_DIR

EVS = {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 2}
SPECIAL = {"hp": 32, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 2}
PASSER = {"species": "Scolipede", "item": "Mental Herb", "ability": "Speed Boost", "nature": "Jolly",
          "moves": ["Iron Defense", "Protect", "Substitute", "Baton Pass"], "evs": EVS, "level": 50}
GHOLDENGO = {"species": "Gholdengo", "item": "Air Balloon", "ability": "Good as Gold", "nature": "Modest",
             "moves": ["Shadow Ball", "Flash Cannon", "Recover", "Nasty Plot"], "evs": SPECIAL, "level": 50}
FLOETTE = {"species": "Floette-Eternal", "item": "Floettite", "ability": "Flower Veil", "nature": "Modest",
           "moves": ["Moonblast", "Substitute", "Synthesis", "Calm Mind"], "evs": SPECIAL, "level": 50}
CHOMP = {"species": "Garchomp", "item": "Life Orb", "ability": "Rough Skin", "nature": "Jolly",
         "moves": ["Earthquake", "Dragon Claw", "Swords Dance", "Protect"], "evs": EVS, "level": 50}
HIPPO = {"species": "Hippowdon", "item": "Leftovers", "ability": "Sand Stream", "nature": "Impish",
         "moves": ["Earthquake", "Yawn", "Slack Off", "Stealth Rock"], "evs": EVS, "level": 50}


def run(code: str, sample: dict) -> dict:
    js = """
const fs = require('fs'), vm = require('vm');
const sandbox = {__dirname: require('path').dirname(process.argv[1]),
    require: n => n === 'readline' ? {createInterface: () => ({on: () => {}})} : require(n),
    process: {argv: ['node', 'worker', process.argv[2]]}, module: {exports: {}}, console};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8') +
    '\\nmodule.exports={setUp,evaluate,tactical,greedy,score,hasDamageSource};', sandbox);
const sample = JSON.parse(process.argv[3]);
const {setUp,evaluate,tactical,greedy,score,hasDamageSource} = sandbox.module.exports;
const format = 'gen9championsbssregmc';
""" + code
    out = subprocess.run(["node", "-e", js, str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR),
                          json.dumps(sample)], capture_output=True, text=True, check=True, timeout=60)
    return json.loads(out.stdout)


def value(result: dict, label: str, matrix: str = "payoff") -> float:
    m = np.array(result[matrix] if matrix == "payoff" else result["byContinuation"][matrix], dtype=float)
    return float(m[result["p1"].index(label)].mean())


def test_tactical_mega_evolves_and_greedy_does_not():
    sample = {"p1": {"team": [FLOETTE, PASSER], "mons": [{"hp": 1.0}, {"hp": 1.0}]},
              "p2": {"team": [HIPPO], "mons": [{"hp": 1.0}]}, "field": {}}
    out = run("const b = setUp(sample, format, 1);"
              "console.log(JSON.stringify({t: tactical(b, b.p1), g: greedy(b, b.p1)}));", sample)
    assert out["t"].endswith("mega") and not out["g"].endswith("mega")


def test_tactical_passes_boosts_when_the_active_cannot_hurt_the_foe():
    sample = {"p1": {"team": [PASSER, GHOLDENGO], "mons": [{"hp": 1.0, "boosts": {"def": 4, "spe": 2}},
                                                             {"hp": 1.0}]},
              "p2": {"team": [CHOMP], "mons": [{"hp": 1.0}]}, "field": {}}
    out = run("const b = setUp(sample, format, 1);"
              "console.log(JSON.stringify({t: tactical(b, b.p1), g: greedy(b, b.p1)}));", sample)
    assert out["t"] == "move 4"  # Baton Pass
    assert out["g"] == "move 1"  # greedy: every move does 0, so the first (Iron Defense) forever


def test_setting_up_then_passing_is_worth_more_when_the_continuation_can_pass():
    sample = {"p1": {"team": [PASSER, GHOLDENGO], "mons": [{"hp": 1.0, "boosts": {"def": 2}}, {"hp": 1.0}]},
              "p2": {"team": [CHOMP], "mons": [{"hp": 1.0}]}, "field": {}}
    out = run("console.log(JSON.stringify(evaluate(sample, format, 2, 4, 0.05, "
              "{continuations: ['greedy', 'tactical']})));", sample)
    assert value(out, "move:irondefense", "tactical") > value(out, "move:irondefense", "greedy")
    both = (value(out, "move:irondefense", "tactical") + value(out, "move:irondefense", "greedy")) / 2
    assert abs(value(out, "move:irondefense") - both) < 1e-9  # the payoff averages the continuations


def test_terminal_scoring_never_ranks_a_position_above_a_win():
    win = {"p1": {"team": [CHOMP, PASSER], "mons": [{"hp": 1.0, "boosts": {"atk": 2}}, {"hp": 1.0}]},
           "p2": {"team": [HIPPO, PASSER], "mons": [{"hp": 0.05}, {"hp": 0}]}, "field": {}}
    out = run("console.log(JSON.stringify(evaluate(sample, format, 1, 0, 0, {scoring: 'terminal'})));", win)
    assert value(out, "move:earthquake") >= 0.95
    assert value(out, "move:protect") < value(out, "move:earthquake")  # a slower win: tie-break only
    assert np.array(out["resolved"])[out["p1"].index("move:earthquake")].min() == 1.0


def test_no_damage_source_left_is_penalised_but_not_a_loss():
    lone = {"p1": {"team": [PASSER, GHOLDENGO], "mons": [{"hp": 1.0}, {"hp": 0}]},
            "p2": {"team": [CHOMP, HIPPO], "mons": [{"hp": 1.0}, {"hp": 0}]}, "field": {}}
    out = run("const b = setUp(sample, format, 1);"
              "console.log(JSON.stringify({ours: hasDamageSource(b, b.p1), theirs: hasDamageSource(b, b.p2),"
              "plain: score(b, b.turn, 0, 'terminal', 0), penalised: score(b, b.turn, 0, 'terminal', 0.4)}));",
              lone)
    assert not out["ours"] and out["theirs"]
    assert out["penalised"] == out["plain"] - 0.4
    assert out["penalised"] > -1  # still not scored as a loss


def test_type_immunity_counts_against_a_damage_source():
    # Gholdengo (Ghost / Steel moves) against a lone Garchomp: Shadow Ball and Flash Cannon both hit
    sample = {"p1": {"team": [GHOLDENGO], "mons": [{"hp": 1.0}]},
              "p2": {"team": [CHOMP], "mons": [{"hp": 1.0}]}, "field": {}}
    out = run("const b = setUp(sample, format, 1); console.log(JSON.stringify(hasDamageSource(b, b.p1)));", sample)
    assert out is True
    normal = dict(CHOMP, species="Gengar", ability="Cursed Body", item="Gengarite",
                  moves=["Shadow Ball", "Sludge Bomb", "Protect", "Will-O-Wisp"])
    tauros = {"species": "Tauros", "item": "Life Orb", "ability": "Intimidate", "nature": "Jolly",
              "moves": ["Body Slam", "Double-Edge", "Protect", "Bulk Up"], "evs": EVS, "level": 50}
    ghost = {"p1": {"team": [tauros], "mons": [{"hp": 1.0}]}, "p2": {"team": [normal], "mons": [{"hp": 1.0}]},
             "field": {}}
    out = run("const b = setUp(sample, format, 1); console.log(JSON.stringify(hasDamageSource(b, b.p1)));", ghost)
    assert out is False  # Normal moves only, into a Ghost type


def test_a_deadline_stops_between_passes_for_every_sample():
    sample = {"p1": {"team": [PASSER, GHOLDENGO], "mons": [{"hp": 1.0}, {"hp": 1.0}]},
              "p2": {"team": [CHOMP], "mons": [{"hp": 1.0}]}, "field": {}}
    req = {"format": "gen9championsbssregmc", "seeds": 50, "rollout": 3, "deadlineMs": 1,
           "continuations": ["greedy", "tactical"], "samples": [sample, sample]}
    out = subprocess.run(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                         input=json.dumps(req) + "\n", capture_output=True, text=True, check=True).stdout
    results = json.loads(out)["results"]
    done = {tuple(r["passes"]) for r in results}
    assert len(done) == 1  # both samples stopped after the same pass
    d, planned = done.pop()
    assert 1 <= d < planned == 100
    assert all(v is not None for r in results for row in r["payoff"] for v in row)
    assert d % 2 == 0  # complete cycles: neither policy disappears when the deadline is short
    assert all(set(r["byContinuation"]) == {"greedy", "tactical"} for r in results)


def test_damage_source_respects_type_conversion_and_ability_immunities():
    gengar = dict(CHOMP, species="Gengar", ability="Cursed Body", item="Gengarite",
                  moves=["Shadow Ball", "Sludge Bomb", "Protect", "Will-O-Wisp"])
    mence = dict(CHOMP, species="Salamence", ability="Intimidate", item="Salamencite",
                 moves=["Double-Edge", "Protect", "Roost", "Dragon Dance"])
    sample = {"p1": {"team": [PASSER, mence], "mons": [{"hp": 1}, {"hp": 1, "mega": True}], "megaUsed": True},
              "p2": {"team": [gengar], "mons": [{"hp": 1}]}, "field": {}}
    assert run("const b=setUp(sample,format,1); console.log(JSON.stringify(hasDamageSource(b,b.p1)));", sample)
    sample["p1"] = {"team": [dict(CHOMP, moves=["Earthquake", "Protect", "Swords Dance", "Substitute"])],
                    "mons": [{"hp": 1}]}
    sample["p2"] = {"team": [dict(gengar, ability="Levitate")], "mons": [{"hp": 1}]}
    assert not run("const b=setUp(sample,format,1); console.log(JSON.stringify(hasDamageSource(b,b.p1)));", sample)
