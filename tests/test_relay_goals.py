"""Receiver thresholds and public weather/Curse controls, with actual engine transitions."""
import pytest
from test_expert_continuation import BULK, CHOMP, DIRGE, FLOETTE, PASSER, side
from test_proposals37 import planned_sample

BAX = dict(CHOMP, species='Baxcalibur', ability='Thermal Exchange', nature='Adamant',
           moves=['Icicle Crash', 'Glaive Rush', 'Ice Shard', 'Earthquake'])
CHAR = dict(CHOMP, species='Charizard-Mega-Y', ability='Drought', item='Charizardite Y', nature='Modest',
            moves=['Flamethrower', 'Solar Beam', 'Roost', 'Air Slash'], evs={'hp': 2, 'spa': 32, 'spe': 32})
UMBREON = dict(FLOETTE, species='Umbreon', ability='Inner Focus', evs=BULK,
               moves=['Snarl', 'Calm Mind', 'Moonlight', 'Baton Pass'])
MEGA_FLOETTE = dict(FLOETTE, item='Floettite')
GOAL = "const {relayStrategy,sunlight}=require('./services/sim-tools/relay_strategy');\n"


def test_one_iron_defense_makes_baxcalibur_a_viable_receiver_and_preserves_priority_handoff():
    sample = {'p1': side([PASSER, MEGA_FLOETTE]), 'p2': side([BAX]), 'field': {}}
    result = planned_sample(sample, GOAL + """
const first=next();advance(first[0]);const hp=b.p1.active[0].hp,second=next();
advance(second[0],'move 3');const pending=next();advance(pending[0],'pass');
console.log(JSON.stringify({choices:[first[1],second[1],pending[1]],hp,
 receiver:b.p1.active[0].species.id,boosts:b.p1.active[0].boosts,
 alive:b.p1.pokemon.every(m=>m.hp>0)}));b.destroy();
""")
    assert result['choices'] == ['move:irondefense', 'move:batonpass', 'switch:floetteeternal']
    assert result['hp'] > 32  # worst Life Orb Ice Shard after +2, before the handoff
    assert result['alive'] and result['boosts']['def'] == 2 and result['boosts']['spe'] == 1


def test_wounded_unboosted_passer_does_not_commit_to_iron_defense_into_priority():
    sample = {'p1': side([PASSER, MEGA_FLOETTE], [{'hp': 32 / 165}, {'hp': 1}]),
              'p2': side([BAX]), 'field': {}}
    result = planned_sample(sample, GOAL + """
console.log(JSON.stringify(relayStrategy(obs.view(b.p1),b.dex)));b.destroy();
""")
    assert result is None


@pytest.mark.parametrize('controller', ['planned', 'expert'])
def test_sun_is_waited_out_with_two_substitutes_not_four(controller):
    sample = {'p1': side([PASSER, UMBREON, MEGA_FLOETTE]), 'p2': side([CHAR]),
              'field': {'weather': 'sunnyday'}}
    result = planned_sample(sample, GOAL + """
const {chooseBatonPublic}=require('./services/sim-tools/public_policy');
obs.append(['|-weather|SunnyDay|[from] ability: Drought|[of] p2a: '+b.p2.active[0].name]);
const labels=[],hp=[],sun=[];
for(let i=0;i<6;i++) {
 const view=obs.view(b.p1),a=CONTROLLER==='planned'?next():chooseBatonPublic(view,'expert',b.dex);
 labels.push(a[1]);hp.push(b.p1.active[0].hp);sun.push(sunlight(view,b.dex));
 if(a[1]==='move:batonpass')break;advance(a[0]);
}
console.log(JSON.stringify({labels,hp,sun}));b.destroy();
""".replace('CONTROLLER', repr(controller)))
    assert result['labels'] == ['move:protect', 'move:substitute', 'move:protect', 'move:substitute',
                                'move:protect', 'move:batonpass']
    assert result['hp'] == [165, 165, 124, 124, 83, 83]
    assert [s['max'] if s else 0 for s in result['sun']] == [5, 4, 3, 2, 1, 0]


def test_unknown_sun_item_keeps_eight_turn_bound_and_reentry_resets_timer():
    sample = {'p1': side([PASSER, MEGA_FLOETTE]), 'p2': side([dict(CHAR, species='Ninetales', item='')]),
              'field': {'weather': 'sunnyday'}}
    result = planned_sample(sample, GOAL + """
const who='p2a: '+b.p2.active[0].name;
obs.append(['|-weather|SunnyDay|[from] ability: Drought|[of] '+who]);
for(let i=0;i<4;i++)obs.append(['|-weather|SunnyDay|[upkeep]']);
const unknown=sunlight(obs.view(b.p1),b.dex);
obs.append(['|-weather|none','|switch|'+who+'|Charizard-Mega-Y, L50|100/100',
 '|-weather|SunnyDay|[from] ability: Drought|[of] '+who]);
const reset=sunlight(obs.view(b.p1),b.dex);
console.log(JSON.stringify({unknown,reset}));b.destroy();
""")
    assert result['unknown'] == {'min': 1, 'max': 4, 'elapsed': 4}
    assert result['reset'] == {'min': 5, 'max': 5, 'elapsed': 0}


def test_fast_receiver_heals_in_sun_before_attempting_calm_mind():
    sample = {'p1': side([UMBREON, MEGA_FLOETTE], [{'hp': 71 / 201, 'boosts': {'spe': 3}}, {'hp': 1}]),
              'p2': side([CHAR]), 'field': {'weather': 'sunnyday'}}
    result = planned_sample(sample, GOAL + """
obs.append(['|-weather|SunnyDay']);
const view=obs.view(b.p1),strategy=relayStrategy(view,b.dex),a=next();advance(a[0]);
console.log(JSON.stringify({goal:strategy.goal,choice:a[1],hp:b.p1.active[0].hp}));b.destroy();
""")
    assert result['choice'] == 'move:moonlight' and result['goal'] == 'recover-before-preparing'
    assert result['hp'] > 0


def test_calm_mind_then_second_hop_remains_available_after_speed_pass():
    sample = {'p1': side([UMBREON, MEGA_FLOETTE], [{'hp': 1, 'boosts': {'spe': 3}}, {'hp': 1}]),
              'p2': side([CHAR]), 'field': {}}
    result = planned_sample(sample, GOAL + """
const a=next();advance(a[0]);const second=next();
console.log(JSON.stringify({choices:[a[1],second[1]],goal:relayStrategy(obs.view(b.p1),b.dex)?.goal}));b.destroy();
""")
    assert result['choices'] == ['move:calmmind', 'move:batonpass']
    assert result['goal'] == 'special-receiver-ready'


@pytest.mark.parametrize('trapped', [False, True])
def test_ghost_curse_requires_a_legal_normal_switch(trapped):
    sample = {'p1': side([UMBREON, MEGA_FLOETTE], [{'hp': .7, 'boosts': {'spe': 3}, 'volatiles': ['curse']},
                                                   {'hp': 1}]), 'p2': side([CHAR]), 'field': {}}
    result = planned_sample(sample, GOAL + """
const view=obs.view(b.p1);view.request.active[0].trapped=TRAPPED;
const s=relayStrategy(view,b.dex);
let remains=null;if(s&&!TRAPPED){advance(s.choice[0]);remains=!!b.p1.active[0].volatiles.curse;}
console.log(JSON.stringify({choice:s?.choice[1]||null,remains}));b.destroy();
""".replace('TRAPPED', str(trapped).lower()))
    assert result['choice'] == (None if trapped else 'switch:floetteeternal')
    if not trapped:
        assert result['remains'] is False


def test_finished_foe_is_attacked_instead_of_starting_a_defensive_goal():
    sample = {'p1': side([MEGA_FLOETTE], [{'hp': .8, 'boosts': {'spe': 3}}]),
              'p2': side([CHAR], [{'hp': .01}]), 'field': {}}
    result = planned_sample(sample, GOAL + """
console.log(JSON.stringify({goal:relayStrategy(obs.view(b.p1),b.dex),choice:next()[1]}));b.destroy();
""")
    assert result['goal'] is None and result['choice'] == 'move:moonblast:mega'


@pytest.mark.parametrize('ability', ['Thermal Exchange', 'Water Veil'])
def test_burn_plan_respects_known_immunity(ability):
    sample = {'p1': side([DIRGE]), 'p2': side([dict(BAX, ability=ability, moves=['Icicle Crash', 'Swords Dance'])],
                                             [{'boosts': {'atk': 2}}]), 'field': {}}
    result = planned_sample(sample, GOAL + """
obs.append(['|-ability|p2a: '+b.p2.active[0].name+'|ABILITY']);
console.log(JSON.stringify(relayStrategy(obs.view(b.p1),b.dex)));b.destroy();
""".replace('ABILITY', ability))
    assert result is None


def test_four_revealed_mimikyu_moves_allow_iron_defense_but_revealed_curse_interrupts_it():
    mim = dict(CHOMP, species='Mimikyu', ability='Disguise', item='Life Orb',
               moves=['Swords Dance', 'Play Rough', 'Shadow Claw', 'Shadow Sneak'])
    sample = {'p1': side([PASSER, MEGA_FLOETTE]), 'p2': side([mim]), 'field': {}}
    result = planned_sample(sample, GOAL + """
const before=relayStrategy(obs.view(b.p1),b.dex);
obs.append(['|move|p2a: '+b.p2.active[0].name+'|Curse|p1a: '+b.p1.active[0].name]);
const after=relayStrategy(obs.view(b.p1),b.dex);
console.log(JSON.stringify({before:before?.goal,after}));b.destroy();
""")
    assert result['before'] == 'prepare-receiver-defence' and result['after'] is None


def test_unaware_will_o_wisp_controls_physical_setup_without_trying_to_match_it():
    mim = dict(CHOMP, species='Mimikyu', ability='Disguise', moves=['Swords Dance', 'Play Rough',
                                                                 'Shadow Claw', 'Shadow Sneak'])
    sample = {'p1': side([DIRGE]), 'p2': side([mim], [{'boosts': {'atk': 2}}]), 'field': {}}
    result = planned_sample(sample, GOAL + """
const s=relayStrategy(obs.view(b.p1),b.dex);const a=next();advance(a[0],'move 1');
console.log(JSON.stringify({goal:s?.goal,choice:a[1],status:b.p2.active[0].status,
 alive:b.p1.active[0].hp>0}));b.destroy();
""")
    assert result['goal'] == 'deny-physical-setup' and result['choice'] == 'move:willowisp'
    assert result['status'] == 'brn' and result['alive']


@pytest.mark.parametrize('constraint', ['pp', 'speed', 'trickroom'])
def test_special_recovery_goal_requires_usable_pp_and_turn_order(constraint):
    sample = {'p1': side([UMBREON, MEGA_FLOETTE], [{'hp': .35, 'boosts': {'spe': 3}}, {'hp': 1}]),
              'p2': side([CHAR]), 'field': {'weather': 'sunnyday'}}
    result = planned_sample(sample, GOAL + """
const view=obs.view(b.p1);
if(CONSTRAINT==='pp')view.request.active[0].moves.find(m=>m.id==='moonlight').pp=0;
if(CONSTRAINT==='speed')view.public.mons.p1[view.public.active.p1].boosts.spe=0;
if(CONSTRAINT==='trickroom')view.history.push('|-fieldstart|move: Trick Room');
console.log(JSON.stringify(relayStrategy(view,b.dex)));b.destroy();
""".replace('CONSTRAINT', repr(constraint)))
    assert result is None


def test_strategy_is_a_root_tiebreak_with_evidence_and_cannot_overrule_a_proven_win():
    from pokesim.search.planner import plan

    goal = {'version': 'public-relay-goals-v1', 'goal': 'recover-before-preparing', 'evidence': {}}
    row = {'p1': ['move:moonlight', 'move:snarl'], 'p2': ['move:flamethrower'],
           'payoff': [[.48], [.5]], 'winSpeed': [[0], [.1]], 'visits': [[10], [10]],
           'rootPlan': 'move:moonlight', 'rootStrategy': goal,
           'terminal_search': {'immediate_wins': []}}
    result = plan([row], pure=True)
    assert result.ranked()[0][0] == 'move:moonlight' and result.status['relay_plan']['strategy'] == goal
    row['payoff'][1][0] = .8
    assert plan([row], pure=True).ranked()[0][0] == 'move:snarl'
    row['payoff'][1][0] = .5
    row['terminal_search']['immediate_wins'] = ['move:snarl']
    assert plan([row], pure=True).ranked()[0][0] == 'move:snarl'


def test_receiver_entry_hazards_are_included_before_calling_the_payload_ready():
    sample = {'p1': side([PASSER, MEGA_FLOETTE],
                         [{'hp': .7, 'boosts': {'def': 2, 'spe': 2}}, {'hp': .6}]),
              'p2': side([BAX]), 'field': {}}
    result = planned_sample(sample, GOAL + """
const view=obs.view(b.p1),before=relayStrategy(view,b.dex);
view.public.conditions.p1.stealthrock=1;
const after=relayStrategy(view,b.dex);
console.log(JSON.stringify({before:before?.goal,after:after?.goal}));b.destroy();
""")
    assert result['before'] == 'receiver-defence-ready'
    assert result.get('after') != 'receiver-defence-ready'


def test_burn_goal_does_not_activate_a_revealed_guts_attacker():
    foe = dict(BAX, species='Conkeldurr', ability='Guts', moves=['Drain Punch', 'Bulk Up'])
    sample = {'p1': side([DIRGE]), 'p2': side([foe], [{'boosts': {'atk': 2}}]), 'field': {}}
    result = planned_sample(sample, GOAL + """
obs.append(['|-ability|p2a: '+b.p2.active[0].name+'|Guts']);
console.log(JSON.stringify(relayStrategy(obs.view(b.p1),b.dex)));b.destroy();
""")
    assert result is None
