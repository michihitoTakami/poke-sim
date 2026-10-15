"""The policy and its learning input must be invariant to hidden opponent sets, on both sides."""
import copy
import json
import subprocess

import pytest
from test_expert_continuation import CHOMP, GHOLDENGO, HIPPO, PASSER, side

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.evaluation.branch_matches import validate_trials

HISTORY = ["|clearpoke", "|poke|p1|Garchomp, L50|", "|poke|p1|Scolipede, L50|",
           "|poke|p2|Hippowdon, L50|", "|poke|p2|Gholdengo, L50|", "|poke|p2|Gengar, L50|",
           "|teampreview|3", "|switch|p1a: Shark|Garchomp, L50|103/202",
           "|switch|p2a: Hippo|Hippowdon, L50|100/100", "|turn|7"]


def run(request):
    out = subprocess.run(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                         input=json.dumps(request) + "\n", capture_output=True, text=True, check=True, timeout=90)
    return json.loads(out.stdout)


def request(samples=None, **options):
    return {"format": "gen9championsbssregmc", "samples": samples or [
        {"p1": side([dict(CHOMP, item=""), PASSER], [{"hp": .51}, {"hp": 1}]),
         "p2": side([HIPPO, GHOLDENGO]), "field": {}}],
        "branches": {"seeds": [0], "maxTurns": 40, "continuations": ["expert/tactical"],
                     "informationScope": "public_observation", "observationLog": HISTORY,
                     "actions": ["move:dragonclaw"], **options}}


def initial(result, world, side_id):
    return next(d for r in result["trials"] if r["world"] == world
                for d in r["trajectory"] if d["side"] == side_id and d["step"] == 0)


def test_p1_input_does_not_include_hidden_opponent_moves_item_stats_or_selected_bench():
    req = request(maxTurns=2)
    other = copy.deepcopy(req["samples"][0])
    other["p2"]["team"][0] = dict(HIPPO, item="Choice Band", ability="Sand Force", nature="Jolly",
                                        moves=["Ice Fang", "Crunch", "Rest", "Whirlwind"],
                                        evs={"hp": 0, "atk": 32, "def": 0, "spa": 0, "spd": 2, "spe": 32})
    other["p2"]["team"][1] = dict(GHOLDENGO, species="Gengar", ability="Cursed Body")
    req["samples"].append(other)
    result = run(req)
    assert initial(result, 0, 'p1') == initial(result, 1, 'p1')
    view = initial(result, 0, 'p1')["observation"]
    assert "choiceband" not in json.dumps(view).lower()
    assert view["public"]["preview"]["p2"] == ["Hippowdon, L50", "Gholdengo, L50", "Gengar, L50"]
    assert len(view["public"]["mons"]["p2"]) == 1  # the hidden selected bench is not revealed
    assert view["public"]["mons"]["p2"]["hippowdon"]["moves"] == []
    assert initial(result, 0, 'p2')["observation"] != initial(result, 1, 'p2')["observation"]


def test_p2_input_cannot_read_the_opponents_private_set_either():
    req = request(maxTurns=2)
    other = copy.deepcopy(req["samples"][0])
    other["p1"]["team"][0] = dict(CHOMP, item="Leftovers", moves=["Dragon Claw", "Fire Fang", "Rest", "Toxic"])
    other["p1"]["team"][1] = GHOLDENGO
    req["samples"].append(other)
    result = run(req)
    assert initial(result, 0, 'p2') == initial(result, 1, 'p2')
    assert initial(result, 0, 'p1')["observation"] != initial(result, 1, 'p1')["observation"]


def test_public_history_hides_exact_hp_and_preserves_source_turn_numbers():
    result = run(request(maxTurns=2))
    p1, p2 = initial(result, 0, 'p1'), initial(result, 0, 'p2')
    assert "103/202" not in json.dumps(p2["observation"])
    assert "50/100g" in json.dumps(p2["observation"])
    assert '/' in p1["observation"]["request"]["side"]["pokemon"][0]["condition"]
    assert p1["observation"]["turn"] == 7
    assert all(d["observation"]["turn"] >= 7 for r in result["trials"] for d in r["trajectory"])
    assert all(not any(h.startswith('|turn|1') and h != '|turn|10' for h in d["observation"]["history"])
               for r in result["trials"] for d in r["trajectory"] if d["observation"]["turn"] < 10)


def test_public_actions_respond_to_revealed_threats():
    # A forced switch policy receives only this observation, never an engine object.
    js = """
const {Dex}=require(process.argv[1]+'/dist/sim');
const {choosePublic}=require(process.argv[2]+'/services/sim-tools/public_policy');
const own=[{ident:'p1: Scolipede',details:'Scolipede, L50',condition:'0 fnt',active:true,
  moves:['protect'],stats:{atk:100,def:100,spa:100,spd:100,spe:100}},
 {ident:'p1: Floette',details:'Floette-Eternal, L50',condition:'173/173',active:false,
  ability:'flowerveil',moves:['moonblast'],stats:{atk:80,def:100,spa:160,spd:140,spe:90}},
 {ident:'p1: Gholdengo',details:'Gholdengo, L50',condition:'175/175',active:false,
  ability:'goodasgold',moves:['shadowball'],stats:{atk:80,def:100,spa:160,spd:100,spe:100}}];
const view={side:'p1',phase:'switch',turn:2,request:{forceSwitch:[true],side:{pokemon:own}},
 public:{active:{p2:'hippowdon'},mons:{p1:{},p2:{hippowdon:{species:'Hippowdon',condition:'100/100',
   moves:[],boosts:{},volatiles:[]}}}}};
const before=choosePublic(view,'expert',Dex);
view.public.mons.p2.hippowdon.moves=['yawn'];
console.log(JSON.stringify([before,choosePublic(view,'expert',Dex)]));
"""
    result = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR), str(ROOT)], check=True,
                            capture_output=True, text=True, timeout=20)
    before, after = json.loads(result.stdout)
    assert before[1] != after[1]
    assert after[1] == 'switch:gholdengo'


def test_terminal_targets_follow_the_observers_perspective_and_root_is_marked():
    req = request(maxTurns=100)
    result = run(req)
    validate_trials(result, 1, [0], ["expert/tactical"])
    assert result["policyVersion"] == "public-reference-v2"
    assert result["trials"][0]["outcome"] in {'win', 'loss', 'draw'}
    trace = result["trials"][0]["trajectory"]
    assert any(not d["rootIntervention"] and d["phase"] == 'move' for d in trace)
    assert all(d["action"] in d["legal"] for d in trace)
    assert all(d["rootIntervention"] == (d["step"] == 0) for d in trace)
    assert trace[0]["target"] == 1 - trace[1]["target"]
    assert all(not any(h.startswith(('|win|', '|tie')) for h in d['observation']['history']) for d in trace)


def test_censored_rollouts_have_no_value_labels():
    result = run(request(maxTurns=1))
    assert result["trials"][0]["outcome"] == 'unfinished'
    assert result["trials"][0]["target"] is None
    assert all(d["target"] is None for d in result["trials"][0]["trajectory"])


@pytest.mark.parametrize('setting', [dict(observationLog=[]), dict(continuations=['expert@4/tactical'])])
def test_public_mode_rejects_missing_history_and_unknown_reference_policies(setting):
    result = run(request(**setting))
    assert 'error' in result or all(r['outcome'] == 'failed' for r in result['trials'])


def test_candidate_order_does_not_change_public_policy_trajectories():
    req = request(actions=['move:dragonclaw', 'move:protect'], maxTurns=10)
    before = run(req)
    req['branches']['actions'].reverse()
    after = run(req)
    assert sorted(before['trials'], key=lambda r: r['action']) == sorted(after['trials'], key=lambda r: r['action'])


def test_baton_pass_preserves_publicly_observed_boosts_and_substitute():
    js = """
const {Dex}=require(process.argv[1]+'/dist/sim');
const {extractChannelMessages}=require(process.argv[1]+'/dist/sim/battle');
const {PublicObservation}=require(process.argv[2]+'/services/sim-tools/public_observation');
const history=['|switch|p1a: Scolipede|Scolipede, L50|100/165','|turn|3',
 '|-boost|p1a: Scolipede|def|4','|-boost|p1a: Scolipede|spe|2','|-start|p1a: Scolipede|Substitute',
 '|-start|p1a: Scolipede|move: Disable|Protect','|move|p1a: Scolipede|Baton Pass|p1a: Scolipede',
 '|switch|p1a: Floette|Floette-Eternal, L50|173/173'];
const observer=new PublicObservation(history,Dex,extractChannelMessages);
console.log(JSON.stringify(observer.state.mons.p1.floette));
"""
    result = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR), str(ROOT)], check=True,
                            capture_output=True, text=True, timeout=20)
    mon = json.loads(result.stdout)
    assert mon['boosts'] == {'def': 4, 'spe': 2}
    assert 'substitute' in mon['volatiles']
    assert 'disable' not in mon['volatiles']


def test_illusion_replace_without_hp_corrects_identity_without_resetting_the_position():
    js = """
const {Dex}=require(process.argv[1]+'/dist/sim');
const {extractChannelMessages}=require(process.argv[1]+'/dist/sim/battle');
const {PublicObservation}=require(process.argv[2]+'/services/sim-tools/public_observation');
const history=['|switch|p2a: Sneasler|Sneasler, L50|70/100','|turn|1',
 '|move|p2a: Sneasler|Close Combat|p1a: Shark',
 '|switch|p2a: Other|Dragalge, L50|100/100',
 '|switch|p2a: Sneasler|Sneasler, L50|100/100',
 '|move|p2a: Sneasler|Shadow Ball|p1a: Shark',
 '|-damage|p2a: Sneasler|40/100','|-boost|p2a: Sneasler|spa|2',
 '|replace|p2a: Zoroark|Zoroark-Hisui, L50'];
const o=new PublicObservation(history,Dex,extractChannelMessages);
console.log(JSON.stringify(o.state));
"""
    result = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR), str(ROOT)], check=True,
                            capture_output=True, text=True, timeout=20)
    state = json.loads(result.stdout)
    assert state['active']['p2'] == 'zoroark'
    real, disguise = state['mons']['p2']['zoroark'], state['mons']['p2']['sneasler']
    assert real['condition'] == '40/100' and real['boosts'] == {'spa': 2}
    assert real['moves'] == ['shadowball']
    assert disguise['condition'] == '70/100' and disguise['moves'] == ['closecombat']
