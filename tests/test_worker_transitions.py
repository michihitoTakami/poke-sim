"""Exercise the worker's simulator boundary, including rejected choices and real switch entry events."""
import json
import subprocess

import pytest

from pokesim.config import ROOT, SHOWDOWN_DIR

EVS = {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 2}
MENCE = {"species": "Salamence", "item": "Salamencite", "ability": "Intimidate", "nature": "Adamant",
         "moves": ["Double-Edge", "Earthquake", "Dragon Dance", "Roost"], "evs": EVS, "level": 50}
CHOMP = {"species": "Garchomp", "item": "Life Orb", "ability": "Rough Skin", "nature": "Jolly",
         "moves": ["Earthquake", "Dragon Claw", "Swords Dance", "Protect"], "evs": EVS, "level": 50}
PASSER = {"species": "Scolipede", "item": "Mental Herb", "ability": "Speed Boost", "nature": "Jolly",
          "moves": ["Iron Defense", "Protect", "Substitute", "Baton Pass"], "evs": EVS, "level": 50}


def worker(code, sample):
    # Do not start the stdin server; export its internal functions from a VM for boundary tests.
    js = """
const fs = require('fs'), vm = require('vm');
const sandbox = {require: n => n === 'readline' ? {createInterface: () => ({on: () => {}})} : require(n),
    process: {argv: ['node', 'worker', process.argv[2]]}, module: {exports: {}}, console};
const source = fs.readFileSync(process.argv[1], 'utf8');
vm.runInNewContext(source + '\\nmodule.exports={setUp,step,evaluate};', sandbox);
const sample = JSON.parse(process.argv[3]);
const {setUp,step,evaluate} = sandbox.module.exports;
const format = 'gen9championsbssregmc';
""" + code
    out = subprocess.run(["node", "-e", js, str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR),
                          json.dumps(sample)], capture_output=True, text=True, check=True, timeout=20)
    return json.loads(out.stdout)


def sample_for_switch(hp=1.0, conditions=None, baton=False):
    return {"p1": {"team": [PASSER, MENCE, CHOMP], "mons": [{"hp": 1.0 if baton else 0,
                    "boosts": {"def": 2}, "volatiles": ["substitute"] if baton else []},
                    {"hp": hp}, {"hp": 1.0}], "conditions": conditions or {}, "incoming": 1,
                    **({"switchMove": "batonpass"} if baton else {})},
            "p2": {"team": [CHOMP], "mons": [{"hp": 1.0}]}, "field": {}}


def test_entry_hazards_can_faint_the_incoming_pokemon():
    result = worker("""
const b = setUp(sample,format,1), mon = b.p1.active[0];
console.log(JSON.stringify({hp:mon.hp, fainted:mon.fainted, request:b.p1.activeRequest}));
""", sample_for_switch(hp=0.1, conditions={"stealthrock": 1}))
    assert result["hp"] == 0 and result["fainted"]
    assert result["request"]["forceSwitch"] == [True]


def test_entry_hazards_and_intimidate_are_applied_once():
    result = worker("""
const b = setUp(sample,format,1), mon = b.p1.active[0];
console.log(JSON.stringify({hp:mon.hp, maxhp:mon.maxhp, attack:b.p2.active[0].boosts.atk}));
""", sample_for_switch(conditions={"stealthrock": 1}))
    assert result["hp"] == result["maxhp"] - result["maxhp"] // 4
    assert result["attack"] == -1


def test_baton_pass_uses_the_simulators_transfer_rules():
    result = worker("""
const b=setUp(sample,format,1), mon=b.p1.active[0];
console.log(JSON.stringify({defense:mon.boosts.def, substitute:!!mon.volatiles.substitute,
                           oldDefense:b.p1.pokemon[1].boosts.def}));
""", sample_for_switch(baton=True))
    assert result == {"defense": 2, "substitute": True, "oldDefense": 0}


def test_an_already_mega_evolved_bench_member_keeps_its_form_on_entry():
    sample = sample_for_switch()
    sample["p1"]["mons"][1]["mega"] = True
    sample["p1"]["megaUsed"] = True
    result = worker("""
const b=setUp(sample,format,1), mon=b.p1.active[0];
console.log(JSON.stringify({species:mon.species.name, ability:mon.ability, canMega:!!mon.canMegaEvo}));
""", sample)
    assert result == {"species": "Salamence-Mega", "ability": "aerilate", "canMega": False}


@pytest.mark.parametrize("injection", [
    "choices=(b,s)=>s.id==='p1' ? [['move 999','invalid']] : original(b,s);",
    "choices=(b,s)=>s.id==='p1' ? [['pass','invalid']] : original(b,s);",
])
def test_rejected_or_stalled_branches_are_not_scored(injection):
    sample = {"p1": {"team": [CHOMP], "mons": [{"hp": 1}]},
              "p2": {"team": [CHOMP], "mons": [{"hp": 1}]}, "field": {}}
    result = worker("""
vm.runInContext("const original=choices; " + """ + json.dumps(injection) + """,sandbox);
console.log(JSON.stringify(sandbox.module.exports.evaluate(sample,format,1,1,0)));
""", sample)
    assert all(v is None for row in result["payoff"] for v in row)
    assert all(v == 1 for row in result["invalid"] for v in row)


def test_a_stalled_request_raises_instead_of_scoring_the_position():
    sample = {"p1": {"team": [CHOMP], "mons": [{"hp": 1}]},
              "p2": {"team": [CHOMP], "mons": [{"hp": 1}]}, "field": {}}
    result = worker("""
const b=setUp(sample,format,1);
b.p1.activeRequest.wait=true;
b.p2.activeRequest.wait=true;
try {step(b,'pass','pass'); console.log(JSON.stringify({error:null}));}
catch(e) {console.log(JSON.stringify({error:e.message}));}
""", sample)
    assert result["error"] == "Branch made no progress"


def test_guard_exhaustion_invalidates_an_unfinished_rollout():
    sample = {"p1": {"team": [CHOMP], "mons": [{"hp": 1}]},
              "p2": {"team": [CHOMP], "mons": [{"hp": 1}]}, "field": {}}
    result = worker("""
vm.runInContext("step=(battle)=>{battle.requestState='switch';};",sandbox);
console.log(JSON.stringify(sandbox.module.exports.evaluate(sample,format,1,1,0)));
""", sample)
    assert all(v is None for row in result["payoff"] for v in row)
    assert all(v == 1 for row in result["invalid"] for v in row)
