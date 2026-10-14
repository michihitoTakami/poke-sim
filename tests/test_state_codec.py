import json
import subprocess

import pytest
from poke_env.battle import Pokemon
from test_worker_transitions import CHOMP, PASSER, worker

from pokesim.state.codec import known_counters
from pokesim.state.tracker import ProtocolTracker

GENGAR = {**CHOMP, "species": "Gengar", "item": "Leftovers", "ability": "Cursed Body",
          "moves": ["Protect", "Disable", "Encore", "Toxic"]}


def test_observable_codec_does_not_invent_hidden_counters():
    mon = Pokemon(gen=9, species="garchomp")
    mon.moved("earthquake")
    mon.item = "choicescarf"
    ours = known_counters(mon, True, own=True)
    theirs = known_counters(mon, True)
    assert ours["ppUsed"]["earthquake"] == 1
    assert ours["choiceMove"] == "earthquake"
    assert "ppUsed" not in theirs and "choiceMove" not in theirs
    assert "sleepRemaining" not in ours and "substituteHp" not in ours
    mon.switch_out({})
    assert "choiceMove" not in known_counters(mon, True, own=True)


@pytest.mark.parametrize("scenario,turns,next_moves", [
    ("choice", [["move 1", "move 1"]], ["move 1", "move 1"]),
    ("disable", [["move 3", "move 1"], ["move 3", "move 2"]], ["move 2", "move 1"]),
    ("encore", [["move 3", "move 1"], ["move 3", "move 3"]], ["move 3", "move 1"]),
    ("substitute", [["move 3", "move 1"]], ["move 1", "move 1"]),
    ("toxic", [["move 3", "move 4"], ["move 3", "move 1"]], ["move 3", "move 1"]),
    ("sleep", [["move 3", "move 4"]], ["move 3", "move 1"]),
])
def test_reconstructed_requests_and_next_turn_match_running_simulator(scenario, turns, next_moves):
    me = {**CHOMP, "item": "Choice Scarf"} if scenario == "choice" else CHOMP
    if scenario == "substitute":
        me = PASSER
    foe = GENGAR
    if scenario == "sleep":
        foe = {**GENGAR, "moves": ["Protect", "Disable", "Encore", "Spore"]}
    if scenario == "substitute":
        foe = {**GENGAR, "moves": ["Tackle", "Disable", "Encore", "Toxic"]}
    sample = {"p1": {"team": [me], "mons": [{}]}, "p2": {"team": [foe], "mons": [{}]}, "field": {}}
    # Oracle is a running simulator, not search's preferred move. Hidden counters below are synthetic test data.
    code = """
const path=require('path');
const {Battle,Teams,PRNG}=require(path.resolve(process.argv[2],'dist/sim'));
const original=new Battle({formatid:format,seed:[1,2,3,4]});
for(const id of ['p1','p2']) original.setPlayer(id,{name:id,team:Teams.pack(sample[id].team)});
original.choose('p1','team 123'); original.choose('p2','team 123');
for(const [a,c] of TURNS) step(original,a,c);
function encode(m) {
 const s={hp:m.hp/m.maxhp,status:m.status,boosts:{...m.boosts},
          pp:Object.fromEntries(m.moveSlots.map(s=>[s.id,s.pp])),volatiles:Object.keys(m.volatiles),
          volatileTurns:Object.fromEntries(Object.entries(m.volatiles).filter(([id,v])=>v.duration!==undefined)
                                         .map(([id,v])=>[id,v.duration]))};
 if(m.lastMove) s.lastMove=m.lastMove.id;
 if(m.volatiles.choicelock) s.choiceMove=m.volatiles.choicelock.move;
 if(m.volatiles.disable) {s.disabledMove=m.volatiles.disable.move;s.disableTurns=m.volatiles.disable.duration;}
 if(m.volatiles.encore) {s.encore=m.volatiles.encore.move;s.encoreTurns=m.volatiles.encore.duration;}
 if(m.volatiles.substitute) s.substituteHp=m.volatiles.substitute.hp;
 if(m.volatiles.stall) {s.stall=Math.log(m.volatiles.stall.counter)/Math.log(3);}
 if(m.status==='tox') s.toxicStage=m.statusState.stage;
 if(m.status==='slp') s.sleepRemaining=m.statusState.time;
 return s;
}
// Specialized conditions need their parameters; restoring bare ids can invoke onStart without lastMove.
for(const id of ['p1','p2']) {
 sample[id].mons=original[id].pokemon.map(encode);
 sample[id].mons.forEach(s=>s.volatiles=s.volatiles.filter(v=>!['choicelock','disable','encore','stall'].includes(v)));
}
const restored=setUp(sample,format,10);
restored.prng=new PRNG(original.prng.getSeed()); restored.turn=original.turn;
function view(b) {
 return ['p1','p2'].map(id=>({mons:b[id].pokemon.map(encode),
    moves:b[id].activeRequest?.active?.[0]?.moves.map(m=>({id:m.id,pp:m.pp,disabled:!!m.disabled})),
    forceSwitch:b[id].activeRequest?.forceSwitch,ended:b.ended,winner:b.winner}));
}
const before=[view(original),view(restored)];
const transitions=[];
for(let i=0;i<3 && !original.ended && !restored.ended;i++) {
 step(original,...NEXT);step(restored,...NEXT);
 transitions.push([view(original),view(restored)]);
}
console.log(JSON.stringify({before,transitions}));
""".replace("TURNS", json.dumps(turns)).replace("NEXT", json.dumps(next_moves))
    result = worker(code, sample)
    state = result["before"][0][0]["mons"][0]
    if scenario in {"choice", "disable", "encore", "substitute"}:
        expected = {"choice": "choicelock", "disable": "disable", "encore": "encore",
                    "substitute": "substitute"}[scenario]
        assert expected in state["volatiles"]
    else:
        assert state["status"] == {"toxic": "tox", "sleep": "slp"}[scenario]
    assert result["before"][0] == result["before"][1]
    for original, restored in result["transitions"]:
        assert original == restored


def test_zero_pp_produces_struggle_and_invalid_pp_is_rejected():
    sample = {"p1": {"team": [CHOMP], "mons": [{"pp": {"earthquake": 0, "dragonclaw": 0,
              "swordsdance": 0, "protect": 0}}]}, "p2": {"team": [CHOMP], "mons": [{}]}}
    result = worker("console.log(JSON.stringify(setUp(sample,format,1).p1.activeRequest.active[0].moves));", sample)
    assert result[0]["id"] == "struggle"
    sample["p1"]["mons"][0]["pp"]["earthquake"] = -1
    with pytest.raises(subprocess.CalledProcessError):
        worker("setUp(sample,format,1);", sample)


def test_protocol_lock_does_not_follow_later_moves_and_ends_on_switch():
    tracker = ProtocolTracker()
    lines = ["|switch|p2a: Garchomp|Garchomp, L50|100/100", "|move|p2a: Garchomp|Earthquake|p1a: Me",
             "|-start|p2a: Garchomp|Disable|Earthquake", "|-start|p2a: Garchomp|Encore",
             "|move|p2a: Garchomp|Dragon Claw|p1a: Me"]
    for line in lines:
        tracker.apply(line.split("|"))
    assert tracker.locks("p2", "Garchomp") == {"disabledMove": "earthquake", "encore": "earthquake"}
    tracker.apply("|-end|p2a: Garchomp|Disable".split("|"))
    assert tracker.locks("p2", "Garchomp") == {"encore": "earthquake"}
    tracker.apply("|switch|p2a: Gengar|Gengar, L50|100/100".split("|"))
    assert tracker.locks("p2", "Garchomp") == {}


def test_champions_active_request_pp_overrides_standard_generation_pp():
    from pokesim.live.session import BattleSession
    from pokesim.search.lookahead import Lookahead

    session = BattleSession([CHOMP])
    session.preview(["Garchomp"])
    session.select([0])
    session.apply(["|switch|p1a: Garchomp|Garchomp, L50|200/200",
                   "|switch|p2a: Garchomp|Garchomp, L50|100/100", "|turn|1"])
    battle = session.decision()
    engine = object.__new__(Lookahead)
    from types import SimpleNamespace

    engine.sampler = SimpleNamespace(base_id=lambda sid: sid)
    side = engine.own_side(battle, {"garchomp": CHOMP})
    assert side["mons"][0]["pp"]["earthquake"] == 12
    result = worker("console.log(JSON.stringify(setUp(sample,format,1).p1.activeRequest.active[0].moves));",
                    {"p1": side, "p2": {"team": [CHOMP], "mons": [{}]}})
    assert result[0]["pp"] == 12
