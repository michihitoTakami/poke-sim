"""A live pivot replaces before a recoil-KO'd foe; it is not a normal turn or a double KO."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_worker_transitions import worker

from pokesim.config import ROOT
from pokesim.live.advisor import Advisor
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id
from pokesim.search.lookahead import Advice, Lookahead

CASE = json.loads((ROOT / 'tests/fixtures/baton-fainted-opponent.json').read_text())


def replay():
    session = BattleSession(CASE['team'])
    session.preview(CASE['preview'])
    session.select([next(i for i, s in enumerate(CASE['team']) if s['species'] == name)
                    for name in CASE['lineup']])
    session.apply(CASE['log'])
    session.battle.parse_request(copy.deepcopy(CASE['request']))
    return session, session.battle


def test_real_recoil_ko_baton_request_is_searched_without_a_fallback(monkeypatch):
    session, battle = replay()
    assert battle.turn == 13 and battle.force_switch
    assert battle.opponent_active_pokemon.fainted and not battle.active_pokemon.fainted
    assert battle.active_pokemon.boosts['spa'] == battle.active_pokemon.boosts['spd'] == 4
    assert {m.species for m in battle.available_switches} == {'gholdengo', 'salamence'}
    advisor = Advisor(CASE['team'], policy=None, lookahead=False, opponent_policy=None,
                      planner={'mode': 'on'})
    advisor.engine = SimpleNamespace()
    searched = []

    def search(b, *args, **kwargs):
        searched.append(b.last_request['rqid'])
        return Advice(['switch:gholdengo', 'switch:salamence'], np.array([1., 0.]), np.array([.6, .4]))

    monkeypatch.setattr(advisor, '_search', search)
    suggestion = advisor.advise(battle, log=session.log)
    assert searched == [33]
    assert suggestion.actions[0] == ('switch:gholdengo', 1.)
    assert 'fallback' not in suggestion.status


def sample():
    _, battle = replay()
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=to_id)
    own = engine.own_side(battle, {to_id(s['species']): s for s in CASE['team']})
    assert own['forceSwitch'] is True and own['switchMove'] == 'batonpass'
    # Known public survivors; these hypothetical sets do not use the later Greninja switch or move.
    foes = [{'species': name, 'level': 50, 'ability': ability, 'nature': 'Modest',
             'moves': ['Protect', move], 'evs': {'hp': 2, 'spa': 32, 'spe': 32}}
            for name, ability, move in [('Venusaur', 'Overgrow', 'Sludge Bomb'),
                                        ('Torkoal', 'Drought', 'Lava Plume'),
                                        ('Greninja', 'Protean', 'Surf')]]
    return {'p1': own, 'p2': {'team': foes, 'mons': [{'hp': 0}, {'hp': 0}, {'hp': .38}]}, 'field': {}}


@pytest.mark.parametrize('receiver,hp', [('gholdengo', 165), ('salamence', 132)])
@pytest.mark.parametrize('snapshot', [False, True])
def test_receiver_then_foe_replacement_survives_search_snapshots(receiver, hp, snapshot):
    result = worker("""
let b=setUp(sample,format,1);
if (sample.snapshot) {
  const {State}=require(process.argv[2]+'/dist/sim/state');
  const restored=State.deserializeBattle(JSON.stringify(State.serializeBattle(b)));
  b.destroy();b=restored;
}
const root={ours:choices(b,b.p1).map(x=>x[1]),foe:choices(b,b.p2).map(x=>x[1])};
const choice=choices(b,b.p1).find(x=>x[1]==='switch:'+sample.receiver)[0];
step(b,choice,'pass');
const entered={hp:b.p1.active[0].hp,boosts:b.p1.active[0].boosts,
  ours:choices(b,b.p1).map(x=>x[1]),foe:choices(b,b.p2).map(x=>x[1])};
step(b,'pass',choices(b,b.p2)[0][0]);
console.log(JSON.stringify({root,entered,phase:b.requestState,active:b.p2.active[0].species.id}));
b.destroy();
""", dict(sample(), receiver=receiver, snapshot=snapshot))
    assert set(result['root']['ours']) == {'switch:gholdengo', 'switch:salamence'}
    assert result['root']['foe'] == ['pass']
    assert result['entered']['hp'] == hp
    assert result['entered']['boosts']['spa'] == result['entered']['boosts']['spd'] == 4
    assert result['entered']['ours'] == ['pass']
    assert result['entered']['foe'] == ['switch:greninja']
    assert result['phase'] == 'move' and result['active'] == 'greninja'


def test_normal_turn_and_simultaneous_double_ko_are_not_live_pivot_phases():
    _, battle = replay()
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=to_id)
    sets = {to_id(s['species']): s for s in CASE['team']}
    request = copy.deepcopy(CASE['request'])
    request.pop('forceSwitch')
    battle.parse_request(request)
    assert 'forceSwitch' not in engine.own_side(battle, sets)
    battle.parse_message('|faint|p1a: Umbreon'.split('|'))
    request = copy.deepcopy(CASE['request'])
    request['side']['pokemon'][0]['condition'] = '0 fnt'
    battle.parse_request(request)
    assert 'forceSwitch' not in engine.own_side(battle, sets)


def test_u_turn_ko_uses_the_same_order_without_copying_boosts():
    world = sample()
    world['p1']['team'][0] = {'species': 'Scizor', 'ability': 'Technician', 'nature': 'Adamant',
                             'level': 50, 'moves': ['U-turn', 'Protect'], 'evs': {'hp': 32, 'atk': 32, 'spe': 2}}
    world['p1']['switchMove'] = 'uturn'
    world['p1']['mons'][0]['lastMove'] = 'uturn'
    result = worker("""
const b=setUp(sample,format,1);
const before=choices(b,b.p2).map(x=>x[1]);
step(b,'switch 2','pass');
console.log(JSON.stringify({before,boosts:b.p1.active[0].boosts,foe:choices(b,b.p2).map(x=>x[1])}));
b.destroy();
""", world)
    assert result['before'] == ['pass']
    assert not any(result['boosts'].values())
    assert result['foe'] == ['switch:greninja']
