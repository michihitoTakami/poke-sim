"""Type-loss protocol -> poke-env -> simulator, compared with an independently running battle."""
import json
import subprocess

import pytest
from test_expert_continuation import CHOMP, side
from test_worker_transitions import worker

from pokesim.live.session import BattleSession
from pokesim.search.lookahead import mon_state

CASES = [
    ('Pawmot', 'Double Shock', 'Thunderbolt', ['???', 'Fighting']),
    ('Regieleki', 'Double Shock', 'Thunderbolt', ['???']),
    ('Arcanine', 'Burn Up', 'Flamethrower', ['???']),
    ('Charizard', 'Burn Up', 'Flamethrower', ['???', 'Flying']),
]


@pytest.mark.parametrize('role', ['p1', 'p2'])
@pytest.mark.parametrize('species,loss,attack,types', CASES)
def test_real_type_loss_round_trip_matches_running_engine(role, species, loss, attack, types):
    source = dict(CHOMP, species=species, item='', ability='Illuminate', moves=[loss, attack, 'Roost'])
    target = dict(CHOMP, species='Blissey', item='', ability='Illuminate',
                  moves=['Splash', 'Earthquake', 'Shadow Ball'])
    sample = {role: side([source, CHOMP]), 'p2' if role == 'p1' else 'p1': side([target]), 'field': {}}
    # This oracle starts a real battle and performs the move. No expected type state is injected.
    original = worker("""
const path=require('path'),{Battle,Teams}=require(path.resolve(process.argv[2],'dist/sim'));
const b=new Battle({formatid:format,seed:[1,2,3,4]});
for(const id of ['p1','p2'])b.setPlayer(id,{name:id,team:Teams.pack(sample[id].team)});
b.choose('p1','team 123');b.choose('p2','team 123');step(b,'move 1','move 1');
const role=ROLE,m=b[role].active[0],foe=b[role==='p1'?'p2':'p1'].active[0];
function damages(b,m,foe){
 b.randomizer=d=>d;
 return [[m,foe,ATTACK],[foe,m,'earthquake'],[foe,m,'shadowball']].map(([a,t,mid])=>{
  const move=b.dex.getActiveMove(mid);move.willCrit=false;
  b.activeMove=move;b.activePokemon=a;b.activeTarget=t;return b.actions.getDamage(a,t,move,true);
 });
}
console.log(JSON.stringify({log:b.log,types:m.getTypes(),damage:damages(b,m,foe),
 state:{hp:m.hp/m.maxhp,pp:Object.fromEntries(m.moveSlots.map(s=>[s.id,s.pp]))}}));b.destroy();
""".replace('ROLE', json.dumps(role)).replace('ATTACK', json.dumps(attack)), sample)
    assert original['types'] == types
    own = [source, CHOMP] if role == 'p1' else [target]
    session = BattleSession(own)
    session.preview([target['species']] if role == 'p1' else [source['species'], CHOMP['species']])
    session.select(list(range(len(own))))
    start = next(i for i, line in enumerate(original['log']) if line.startswith('|teampreview')) + 1
    session.apply(original['log'][start:])
    mon = session.battle.active_pokemon if role == 'p1' else session.battle.opponent_active_pokemon
    encoded = mon_state(mon, True, own=role == 'p1')
    assert encoded['types'] == types
    sample[role]['mons'][0] = dict(original['state'], **encoded)
    result = worker("""
const b=setUp(sample,format,1),role=ROLE;
let m=b[role].active[0],foe=b[role==='p1'?'p2':'p1'].active[0];
b.randomizer=d=>d;
const damage=[[m,foe,ATTACK],[foe,m,'earthquake'],[foe,m,'shadowball']].map(([a,t,mid])=>{
 const move=b.dex.getActiveMove(mid);move.willCrit=false;
 b.activeMove=move;b.activePokemon=a;b.activeTarget=t;return b.actions.getDamage(a,t,move,true);
});
const before=m.getTypes(),hp=foe.hp;
step(b,'move 1','move 1');
const repeated={hp:foe.hp,failed:b.log.some(l=>l.includes('|-fail|')&&l.includes('move: '))};
step(b,role==='p1'?'switch 2':'move 1',role==='p2'?'switch 2':'move 1');
step(b,role==='p1'?'switch 2':'move 1',role==='p2'?'switch 2':'move 1');
console.log(JSON.stringify({before,damage,hp,repeated,after:b[role].active[0].getTypes()}));b.destroy();
""".replace('ROLE', json.dumps(role)).replace('ATTACK', json.dumps(attack)), sample)
    assert result['before'] == types
    assert result['damage'] == original['damage']
    assert result['damage'][2] > 0  # Typeless is not Normal: Ghost remains able to hit it.
    assert result['repeated'] == {'hp': result['hp'], 'failed': True}
    assert result['after'] == {'Pawmot': ['Electric', 'Fighting'], 'Regieleki': ['Electric'],
                               'Arcanine': ['Fire'], 'Charizard': ['Fire', 'Flying']}[species]


def test_burn_up_roost_ground_hit_matches_the_running_engine():
    source = dict(CHOMP, species='Charizard', item='', ability='Blaze', moves=['Burn Up', 'Roost'])
    target = dict(CHOMP, species='Blissey', item='', ability='Illuminate', moves=['Splash', 'Earthquake'])
    result = worker("""
const path=require('path'),{Battle,Teams,PRNG}=require(path.resolve(process.argv[2],'dist/sim'));
const original=new Battle({formatid:format,seed:[1,2,3,4]});
for(const id of ['p1','p2'])original.setPlayer(id,{name:id,team:Teams.pack(sample[id].team)});
original.choose('p1','team 123');original.choose('p2','team 123');step(original,'move 1','move 1');
original.p1.active[0].hp=Math.floor(original.p1.active[0].maxhp/2);
for(const id of ['p1','p2'])sample[id].mons=original[id].pokemon.map(m=>({hp:m.hp/m.maxhp,types:m.getTypes()}));
const restored=setUp(sample,format,10);
restored.prng=new PRNG(original.prng.getSeed());restored.turn=original.turn;
for(const b of [original,restored]){
 const getDamage=b.actions.getDamage;
 b.actions.getDamage=function(source,target,move,...args){
  if(move.id==='earthquake')b.typeLossProbe={types:target.getTypes(),grounded:target.isGrounded()};
  return getDamage.call(this,source,target,move,...args);
 };
}
step(original,'move 2','move 2');step(restored,'move 2','move 2');
console.log(JSON.stringify({original:original.p1.active[0].hp,restored:restored.p1.active[0].hp,
 probes:[original.typeLossProbe,restored.typeLossProbe],
 maxhp:original.p1.active[0].maxhp,groundHit:original.log.some(l=>l.startsWith('|-damage|p1'))}));
original.destroy();restored.destroy();
""", {'p1': side([source]), 'p2': side([target]), 'field': {}})
    assert result['groundHit'] and result['original'] < result['maxhp']
    assert result['original'] == result['restored']
    assert result['probes'] == [{'types': ['???'], 'grounded': True}] * 2


@pytest.mark.parametrize('species,loss,attack,types', CASES)
def test_public_damage_loses_stab_and_preserves_the_remaining_type(species, loss, attack, types):
    source = dict(CHOMP, species=species, item='', ability='Illuminate', moves=[loss, attack])
    target = dict(CHOMP, species='Blissey', item='', ability='Illuminate', moves=['Splash'])
    result = worker("""
const {PublicObservation}=require('./services/sim-tools/public_observation');
const {estimateDamage}=require('./services/sim-tools/public_damage');
const {extractChannelMessages}=require(process.argv[2]+'/dist/sim/battle');
const b=setUp(sample,format,1),obs=new PublicObservation(b.log,b.dex,extractChannelMessages);
function damage(){
 const view=obs.view(b.p1),me=view.request.side.pokemon.find(m=>m.active),foe=view.public.mons.p2.blissey;
 return [estimateDamage(view,b.dex,me,foe,b.dex.moves.get(ATTACK)),
         estimateDamage(view,b.dex,foe,me,b.dex.moves.get('earthquake')),
         estimateDamage(view,b.dex,foe,me,b.dex.moves.get('shadowball'))];
}
const before=damage();obs.append(['|-start|p1a: '+b.p1.active[0].name+'|typechange|'+TYPES.join('/')]);
console.log(JSON.stringify({before,after:damage(),types:obs.view(b.p1).public.mons.p1[obs.state.active.p1].types}));
b.destroy();
""".replace('ATTACK', json.dumps(attack)).replace('TYPES', json.dumps(types)),
                    {'p1': side([source]), 'p2': side([target]), 'field': {}})
    assert result['types'] == types
    assert result['after'][0] == pytest.approx(result['before'][0] / 1.5)
    assert result['after'][2] > 0
    if 'Flying' in types:
        assert result['after'][1] == 0
    else:
        assert result['after'][1] == pytest.approx(result['before'][1] / 2)


@pytest.mark.parametrize('types', [['nonsense'], ['Three_Question_Marks'], ['Stellar'], ['stellar'], [], [None]])
def test_type_loss_support_does_not_accept_other_invalid_type_states(types):
    sample = {'p1': side([CHOMP], [{'hp': 1, 'types': types}]), 'p2': side([CHOMP])}
    with pytest.raises(subprocess.CalledProcessError):
        worker('setUp(sample,format,1);', sample)
