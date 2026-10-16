import copy
import json
import subprocess

import pytest
from test_expert_continuation import CHOMP, GHOLDENGO, side

from pokesim.config import ROOT, SHOWDOWN_DIR

NORMAL = dict(CHOMP, species='Blissey', ability='Natural Cure', item='', moves=['Splash'])


def roots(samples, probe=None):
    code = r"""
const fs=require('fs'),vm=require('vm');
const sandbox={__dirname:process.cwd()+'/services/sim-tools',
 require:n=>n==='readline'?{createInterface:()=>({on:()=>{}})}:require(n),
 process:{argv:['node','worker',process.argv[1]]},module:{exports:{}}};
vm.runInNewContext(fs.readFileSync('./services/sim-tools/lookahead.js','utf8')+
 '\nmodule.exports={setUp,step,choices,State,extractChannelMessages};',sandbox);
const {Search}=require('./services/sim-tools/terminal_search');
const s=new Search({format:'gen9championsbssregmc',observationLog:['|turn|1'],
 samples:JSON.parse(fs.readFileSync(0,'utf8'))},sandbox.module.exports);
console.log(JSON.stringify({options:s.states.map(st=>st.p1.map(x=>x[1])),filter:s.actionFilter}));
"""
    if probe:
        code = code[:code.index('console.log(JSON.stringify({options:')] + probe
    result = subprocess.run(['node', '-e', code, str(SHOWDOWN_DIR)], cwd=ROOT,
                            input=json.dumps(samples), text=True, capture_output=True, check=True, timeout=10)
    return json.loads(result.stdout)


def sample(ours=None, foe=None, state=None, field=None):
    return {'p1': side([ours or GHOLDENGO], [state or {'hp': 1}]), 'p2': side([foe or NORMAL]), 'field': field or {}}


def test_full_hp_heal_and_immune_attack_and_capped_setup_are_removed_together():
    result = roots([sample(state={'hp': 1, 'boosts': {'spa': 6}})])
    assert result['options'] == [['move:flashcannon']]
    excluded = {r['label']: r['reasons'] for r in result['filter']['excluded']}
    assert excluded['move:recover'] == ['full-hp-heal-before-any-damage']
    assert excluded['move:nastyplot'] == ['boosts-already-capped']
    assert excluded['move:shadowball'] == ['immune-to-every-sampled-reply']


@pytest.mark.parametrize('case', ['faster', 'priority', 'tie', 'quickclaw', 'laggingtail'])
def test_full_hp_heal_is_kept_when_damage_can_arrive_before_it(case):
    ours, foe = dict(GHOLDENGO), dict(CHOMP, item='', moves=['Dragon Claw'], evs=dict(CHOMP['evs'], hp=2, spe=32))
    if case == 'priority':
        foe.update(species='Blissey', moves=['Aqua Jet'])
    elif case == 'tie':
        foe = dict(GHOLDENGO, item='', moves=['Shadow Ball'])
    elif case == 'quickclaw':
        foe.update(species='Blissey', item='Quick Claw')
    elif case == 'laggingtail':
        ours['item'] = 'Lagging Tail'
        foe['species'] = 'Blissey'
    result = roots([sample(ours, foe)])
    assert 'move:recover' in result['options'][0]


def test_trick_room_order_is_used_in_the_heal_filter():
    foe = dict(CHOMP, item='', moves=['Dragon Claw'], evs=dict(CHOMP['evs'], hp=2, spe=32))
    result = roots([sample(foe=foe, field={'trickroom': 3})])
    assert 'move:recover' not in result['options'][0]  # slower Gholdengo recovers before any hit


def test_damaged_heal_and_unsaturated_setup_are_kept():
    result = roots([sample(state={'hp': .5, 'boosts': {'spa': 4}})])
    assert {'move:recover', 'move:nastyplot'} <= set(result['options'][0])


@pytest.mark.parametrize('move,ability,species', [('Snarl', 'Infiltrator', 'Dragapult'),
    ('Haze', 'Prankster', 'Murkrow')])
def test_capped_setup_is_kept_if_opponent_can_lower_or_reset_it_first(move, ability, species):
    foe = dict(CHOMP, species=species, ability=ability, item='', moves=[move])
    result = roots([sample(foe=foe, state={'hp': 1, 'boosts': {'spa': 6}})])
    assert 'move:nastyplot' in result['options'][0]


def test_capped_attack_boost_is_kept_when_a_switch_can_intimidate():
    ours = dict(CHOMP, item='', moves=['Swords Dance', 'Dragon Claw'])
    pos = sample(ours=ours, state={'hp': 1, 'boosts': {'atk': 6}})
    pos['p2'] = side([NORMAL, dict(CHOMP, species='Salamence', ability='Intimidate')])
    assert 'move:swordsdance' in roots([pos])['options'][0]


def test_contrary_does_not_treat_a_positive_cap_as_a_failed_boost():
    ours = dict(GHOLDENGO, ability='Contrary')
    assert 'move:nastyplot' in roots([sample(ours, state={'hp': 1, 'boosts': {'spa': 6}})])['options'][0]
    assert 'move:nastyplot' not in roots([sample(ours, state={'hp': 1, 'boosts': {'spa': -6}})])['options'][0]


def test_attack_is_kept_if_a_possible_switch_target_is_vulnerable():
    pos = sample()
    pos['p2'] = side([NORMAL, dict(GHOLDENGO)])
    assert 'move:shadowball' in roots([pos])['options'][0]


def test_unknown_worlds_share_one_action_set_and_one_effective_world_is_enough_to_keep_a_move():
    a, b = sample(), sample(foe=GHOLDENGO)
    result = roots([a, b])
    assert result['options'][0] == result['options'][1]
    assert 'move:shadowball' in result['options'][0]


@pytest.mark.parametrize('bypass', [False, True])
def test_ability_immunity_respects_mold_breaker(bypass):
    ours = dict(CHOMP, ability='Mold Breaker' if bypass else 'Rough Skin', item='', moves=['Fire Fang', 'Dragon Claw'])
    foe = dict(GHOLDENGO, species='Chandelure', ability='Flash Fire', item='', moves=['Splash'])
    result = roots([sample(ours, foe)])
    assert ('move:firefang' in result['options'][0]) == bypass


def test_scrappy_can_hit_a_ghost_and_good_as_gold_blocks_encore():
    ours = dict(CHOMP, ability='Scrappy', moves=['Double-Edge', 'Encore', 'Earthquake'])
    foe = dict(GHOLDENGO, moves=['Nasty Plot'])
    result = roots([sample(ours, foe)])
    assert 'move:doubleedge' in result['options'][0]
    assert 'move:encore' not in result['options'][0]


def test_a_mega_activation_is_not_discarded_with_a_capped_setup():
    ours = dict(CHOMP, species='Salamence', ability='Intimidate', item='Salamencite',
                moves=['Dragon Dance', 'Dragon Claw'])
    result = roots([sample(ours, state={'hp': 1, 'boosts': {'atk': 6, 'spe': 6}})])
    assert 'move:dragondance' not in result['options'][0]
    assert 'move:dragondance:mega' in result['options'][0]


def test_never_remove_all_legal_actions_or_forced_switch_choices():
    pos = sample(ours=dict(GHOLDENGO, moves=['Shadow Ball']))
    result = roots([pos])
    assert result['options'] == [['move:shadowball']] and result['filter']['all_no_effect']
    pos = copy.deepcopy(pos)
    pos['p1'] = side([GHOLDENGO, CHOMP], [{'hp': 0}, {'hp': 1}])
    assert roots([pos])['options'] == [['switch:garchomp']]


def test_one_deep_win_does_not_discard_the_open_position_anchor():
    probe = """
const cell=s.states[0].cells[0][0];
s.backup({cell,trace:[],initial:true,ended:false,turns:1},.6);
s.backup({cell,trace:[],initial:false,ended:true,turns:4},1);
console.log(JSON.stringify({value:s.reply().results[0].payoff[0][0],horizon:s.maxTurns}));
"""
    result = roots([sample()], probe)
    assert result['value'] == pytest.approx(.36)  # anchor .2 retained with 4 pseudo-observations, not replaced by +1
    assert result['horizon'] == 8


def test_waiting_for_a_residual_ko_is_a_useful_passive_option():
    pos = sample(state={'hp': 1, 'boosts': {'spa': 6}})
    pos['p2']['mons'][0] = {'hp': .01, 'status': 'psn'}
    result = roots([pos])
    assert result['filter']['passive_wait'] == 'move:recover'
    assert 'move:recover' in result['options'][0]
    assert 'move:nastyplot' not in result['options'][0]


@pytest.mark.parametrize('move,species,ability', [
    ('Will-O-Wisp', 'Chandelure', 'Flash Fire'), ('Toxic', 'Metagross', 'Clear Body'),
    ('Thunder Wave', 'Garchomp', 'Rough Skin'), ('Spore', 'Venusaur', 'Overgrow')])
def test_obvious_status_immunities_are_removed(move, species, ability):
    ours = dict(GHOLDENGO, moves=[move, 'Shadow Ball', 'Flash Cannon'])
    foe = dict(CHOMP, species=species, ability=ability, item='', moves=['Splash'])
    label = 'move:' + ''.join(c for c in move.lower() if c.isalnum())
    assert label not in roots([sample(ours, foe)])['options'][0]


def test_corrosion_can_poison_a_steel_type():
    ours = dict(GHOLDENGO, ability='Corrosion', moves=['Toxic', 'Shadow Ball'])
    foe = dict(NORMAL, species='Metagross', ability='Clear Body')
    assert 'move:toxic' in roots([sample(ours, foe)])['options'][0]


def test_full_hp_roost_fails_without_grounding_and_can_be_pruned():
    from test_worker_transitions import worker

    ours = dict(CHOMP, species='Salamence', ability='Intimidate', item='', moves=['Roost', 'Dragon Claw'])
    pos = sample(ours)
    log = worker("""
const b=setUp(sample,format,1),at=b.log.length;
step(b,'move roost','move splash');console.log(JSON.stringify(b.log.slice(at)));
""", pos)
    assert any(line.startswith('|-fail|') and line.endswith('|heal') for line in log)
    assert not any('|-singleturn|' in line and 'Roost' in line for line in log)
    assert 'move:roost' not in roots([pos])['options'][0]


def test_a_two_turn_invulnerability_move_is_not_reduced_to_its_immune_hit():
    ours = dict(CHOMP, item='', moves=['Dig', 'Fire Fang'])
    foe = dict(NORMAL, species='Togekiss', ability='Serene Grace')
    # Dig can still avoid an attack or wait for residual damage, even though the target is Flying.
    assert 'move:dig' in roots([sample(ours, foe)])['options'][0]
