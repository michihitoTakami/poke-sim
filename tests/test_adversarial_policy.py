"""The opponent must execute concrete relay-breaking lines under public observations."""
import pytest
from test_expert_continuation import CHOMP, GHOLDENGO, PASSER, side
from test_proposals37 import planned_sample
from test_relay_goals import UMBREON

POLICY = "const {chooseCounterplay}=require('./services/sim-tools/counterplay_policy');\n"


@pytest.mark.parametrize('species,ability,setup,attacks', [
    ('Gyarados', 'Intimidate', 'Dragon Dance', ['Waterfall', 'Earthquake', 'Ice Fang']),
    ('Blastoise', 'Torrent', 'Shell Smash', ['Water Pulse', 'Ice Beam', 'Dark Pulse']),
])
def test_physical_and_shell_smash_opponents_punish_a_pure_support_passer(species, ability, setup, attacks):
    source = dict(CHOMP, species=species, ability=ability, item='', moves=[setup, *attacks])
    result = planned_sample({'p1': side([source]), 'p2': side([PASSER]), 'field': {}}, POLICY + """
const a=chooseCounterplay(obs.view(b.p1),b.dex);advance(a[0],'move 2');
console.log(JSON.stringify({choice:a[1],boosts:b.p1.active[0].boosts}));b.destroy();
""")
    assert result['choice'] == 'move:' + setup.lower().replace(' ', '')
    assert result['boosts']['spe'] > 0


def test_roar_bypasses_a_substitute_after_setting_rocks():
    source = dict(CHOMP, species='Archaludon', ability='Stamina', item='',
                  moves=['Stealth Rock', 'Roar', 'Body Press', 'Flash Cannon'])
    sample = {'p1': side([source]), 'p2': side([PASSER, GHOLDENGO],
              [{'hp': 1, 'boosts': {'def': 2}, 'volatiles': ['substitute']}, {'hp': 1}]), 'field': {}}
    result = planned_sample(sample, POLICY + """
const first=chooseCounterplay(obs.view(b.p1),b.dex);advance(first[0],'move 2');
const second=chooseCounterplay(obs.view(b.p1),b.dex);advance(second[0],'move 2');
console.log(JSON.stringify({choices:[first[1],second[1]],receiver:b.p2.active[0].species.id,
 hp:b.p2.active[0].hp,maxhp:b.p2.active[0].maxhp}));b.destroy();
""")
    assert result['choices'] == ['move:stealthrock', 'move:roar']
    assert result['receiver'] == 'gholdengo' and result['hp'] < result['maxhp']


def test_ghost_curse_bypasses_a_substitute_and_costs_real_hp():
    source = dict(CHOMP, species='Mimikyu', ability='Disguise', item='',
                  moves=['Curse', 'Play Rough', 'Shadow Claw', 'Shadow Sneak'])
    sample = {'p1': side([source]), 'p2': side([UMBREON],
              [{'hp': 1, 'boosts': {'spa': 2, 'spd': 2}, 'volatiles': ['substitute']}]), 'field': {}}
    result = planned_sample(sample, POLICY + """
const a=chooseCounterplay(obs.view(b.p1),b.dex);advance(a[0],'move 1');
console.log(JSON.stringify({choice:a[1],curse:!!b.p2.active[0].volatiles.curse,
 hp:b.p1.active[0].hp,maxhp:b.p1.active[0].maxhp}));b.destroy();
""")
    assert result['choice'] == 'move:curse' and result['curse']
    assert 0 < result['hp'] <= result['maxhp'] / 2


@pytest.mark.parametrize('item,locked', [('', True), ('Mental Herb', False)])
def test_encore_bypasses_a_substitute_and_respects_mental_herb(item, locked):
    source = dict(CHOMP, species='Gengar', ability='Cursed Body', item='Choice Scarf',
                  moves=['Encore', 'Shadow Ball', 'Protect', 'Substitute'])
    sample = {'p1': side([source]), 'p2': side([dict(PASSER, item=item)],
              [{'hp': 1, 'lastMove': 'protect', 'volatiles': ['substitute']}]), 'field': {}}
    result = planned_sample(sample, POLICY + """
obs.append(['|move|p2a: '+b.p2.active[0].name+'|Protect|p2a: '+b.p2.active[0].name]);
const a=chooseCounterplay(obs.view(b.p1),b.dex);advance(a[0],'move 1');
console.log(JSON.stringify({choice:a[1],encore:b.p2.active[0].volatiles.encore?.move,
 item:b.p2.active[0].item}));b.destroy();
""")
    assert result['choice'] == 'move:encore' and result['item'] == ''
    assert result.get('encore') == ('protect' if locked else None)


def test_a_wounded_attacker_does_not_shell_smash_into_a_lethal_hit():
    source = dict(CHOMP, species='Blastoise', ability='Torrent', item='',
                  moves=['Shell Smash', 'Water Pulse', 'Ice Beam', 'Dark Pulse'])
    result = planned_sample({'p1': side([source], [{'hp': .1}]), 'p2': side([CHOMP]), 'field': {}}, POLICY + """
console.log(JSON.stringify(chooseCounterplay(obs.view(b.p1),b.dex)[1]));b.destroy();
""")
    assert result != 'move:shellsmash'


def test_known_good_as_gold_still_blocks_targeted_status():
    source = dict(CHOMP, species='Gengar', ability='Cursed Body', item='',
                  moves=['Curse', 'Encore', 'Taunt', 'Shadow Ball'])
    result = planned_sample({'p1': side([source]), 'p2': side([GHOLDENGO]), 'field': {}}, POLICY + """
obs.append(['|move|p2a: '+b.p2.active[0].name+'|Nasty Plot|p2a: '+b.p2.active[0].name]);
console.log(JSON.stringify(chooseCounterplay(obs.view(b.p1),b.dex)[1]));b.destroy();
""")
    assert result == 'move:shadowball'
