"""Equipment, observed formes and cumulative faints must survive the public-state boundary."""
import json
from types import SimpleNamespace

import pytest
from test_worker_transitions import CHOMP, worker

from pokesim.config import ROOT
from pokesim.live.session import BattleSession
from pokesim.search.lookahead import Lookahead, mon_state, revealed, to_id


def session_for(sets):
    session = BattleSession(sets)
    session.preview(['Garchomp', 'Basculegion', 'Lucario'])
    session.select(list(range(len(sets))))
    return session


@pytest.mark.parametrize('species,item,event', [
    ('Gholdengo', 'Air Balloon', '|-enditem|p1a: Gholdengo|Air Balloon'),
    ('Skeledirge', 'Sitrus Berry', '|-enditem|p1a: Skeledirge|Sitrus Berry|[eat]'),
    ('Scolipede', 'Mental Herb', '|-enditem|p1a: Scolipede|Mental Herb'),
])
def test_consumed_items_remain_empty_through_requests_and_the_simulator(species, item, event):
    sets = [s for s in json.loads((ROOT / 'data/my_teams/baton.json').read_text()) if s['species'] == species]
    session = session_for(sets)
    name = session.info[0]['name']
    session.apply([f'|switch|p1a: {name}|{species}, L50|100/200',
                   '|switch|p2a: Garchomp|Garchomp, L50|100/100', '|turn|1',
                   event, '|turn|2'])
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=lambda sid: sid)
    for _ in range(2):
        battle = session.decision()
        assert not battle.active_pokemon.item
        side = engine.own_side(battle, {to_id(species): sets[0]})
        assert side['mons'][0]['item'] == ''
        result = worker("""
const b=setUp(sample,format,1), m=b.p1.active[0];
const damage=b.actions.getDamage(b.p2.active[0],m,'earthquake');
console.log(JSON.stringify({item:m.item,damage}));
b.destroy();
""", {'p1': side, 'p2': {'team': [dict(CHOMP, item='')], 'mons': [{}]}})
        assert result['item'] == ''
        if species == 'Gholdengo':
            assert result['damage'] > 0
    assert sets[0]['item'] == item


def test_current_tricked_item_overrides_the_team_sheet_on_the_bench():
    sample = {'p1': {'team': [CHOMP, dict(CHOMP, item='Air Balloon')],
                     'mons': [{}, {'item': 'choicescarf'}], 'incoming': 1},
              'p2': {'team': [CHOMP], 'mons': [{}]}}
    result = worker("""
const b=setUp(sample,format,1), m=b.p1.active[0];
console.log(JSON.stringify({item:m.item,speed:m.getStat('spe'),base:m.storedStats.spe}));
""", sample)
    assert result['item'] == 'choicescarf'
    assert result['speed'] == int(result['base'] * 1.5)


@pytest.mark.parametrize('species,ability,move', [
    ('Basculegion', 'Adaptability', 'Last Respects'),
    ('Kingambit', 'Supreme Overlord', 'Kowtow Cleave'),
])
def test_damage_after_teammate_faints_matches_an_independent_running_battle(species, ability, move):
    attacker = dict(CHOMP, species=species, ability=ability, moves=[move], item='')
    target = dict(CHOMP, species='Snorlax', ability='Immunity', moves=['Splash'], item='')
    sample = {'p1': {'team': [attacker, CHOMP, CHOMP], 'mons': [{}, {'hp': 0}, {'hp': 0}]},
              'p2': {'team': [target], 'mons': [{}]}}
    result = worker("""
const path=require('path'),{Battle,Teams}=require(path.resolve(process.argv[2],'dist/sim'));
// Independent history: two teammates really faint, then the attacker enters.
const original=new Battle({formatid:format,seed:[1,2,3,4]});
original.setPlayer('p1',{name:'p1',team:Teams.pack([sample.p1.team[1],sample.p1.team[2],sample.p1.team[0]])});
original.setPlayer('p2',{name:'p2',team:Teams.pack(sample.p2.team)});
original.choose('p1','team 123');original.choose('p2','team 123');
for(let k=0;k<2;k++) {
 original.p1.active[0].faint(); original.faintMessages(); original.checkFainted();
 original.makeRequest('switch'); original.choose('p1','switch '+(k+2)); original.choose('p2','pass');
}
const restored=setUp(sample,format,1);
function damage(b) {
 b.randomizer=d=>d;const a=b.p1.active[0],t=b.p2.active[0];
 const m=b.dex.getActiveMove(a.moveSlots[0].id);m.willCrit=false;
 b.activeMove=m;b.activePokemon=a;b.activeTarget=t;
 return b.actions.getDamage(a,t,m,true);
}
console.log(JSON.stringify({original:damage(original),restored:damage(restored),
 originalFaints:original.p1.totalFainted,restoredFaints:restored.p1.totalFainted}));
original.destroy();restored.destroy();
""", sample)
    assert result['originalFaints'] == result['restoredFaints'] == 2
    assert result['original'] == result['restored']


@pytest.mark.parametrize('forme,stone,ability', [
    ('Starmie-Mega', 'Starminite', 'hugepower'),
    ('Lucario-Mega-Z', 'Lucarionite Z', 'auraguard'),
    ('Charizard-Mega-X', 'Charizardite X', 'toughclaws'),
    ('Charizard-Mega-Y', 'Charizardite Y', 'drought'),
])
def test_observed_mega_is_preserved_even_if_the_sample_had_another_item(forme, stone, ability):
    session = session_for([CHOMP])
    species = forme.split('-Mega')[0]
    session.apply(['|switch|p1a: Garchomp|Garchomp, L50|200/200',
                   f'|switch|p2a: Foe|{species}, L50|100/100', '|turn|1',
                   f'|detailschange|p2a: Foe|{forme}, L50',
                   f'|-mega|p2a: Foe|{species}|{stone}', '|turn|2'])
    mon = session.decision().opponent_active_pokemon
    state = mon_state(mon, True)
    assert state['forme'] == forme
    assert revealed(mon)[1] == to_id(stone)
    result = worker("""
const b=setUp(sample,format,1), m=b.p2.active[0];
console.log(JSON.stringify({species:m.species.name,ability:m.ability,canMega:!!m.canMegaEvo,
 stats:m.species.baseStats}));b.destroy();
""", {'p1': {'team': [CHOMP], 'mons': [{}]}, 'p2': {
        'team': [dict(CHOMP, species=species, item='Leftovers')], 'mons': [state], 'megaUsed': True}})
    assert result['species'] == forme
    assert result['ability'] == ability
    assert not result['canMega']
    assert result['stats'] == mon.base_stats


def test_cumulative_faints_survive_a_revived_member_and_switches():
    session = session_for([CHOMP])
    session.apply(['|switch|p1a: Garchomp|Garchomp, L50|200/200',
                   '|switch|p2a: Foe|Garchomp, L50|100/100', '|turn|1',
                   '|faint|p2a: Foe', '|-heal|p2: Foe|50/100|[from] move: Revival Blessing',
                   '|switch|p2a: Foe|Garchomp, L50|50/100', '|faint|p2a: Foe'])
    assert session.battle.state_tracker.total_fainted == {'p2': 2}
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=lambda sid: sid, rng=SimpleNamespace(shuffle=lambda x: None),
                                     sample=lambda *a: CHOMP)
    assert engine.opponent_side(session.battle)['totalFainted'] == 2


def test_broken_illusion_does_not_invent_a_fainted_selected_member():
    session = BattleSession([CHOMP])
    session.preview(['Starmie', 'Zoroark-Hisui', 'Garchomp'])
    session.select([0])
    session.apply(['|switch|p1a: Garchomp|Garchomp, L50|200/200',
                   '|switch|p2a: Starmie|Starmie, L50|100/100', '|turn|1',
                   '|move|p2a: Starmie|Shadow Ball|p1a: Garchomp',
                   '|-boost|p2a: Starmie|spa|2', '|-enditem|p2a: Starmie|Focus Sash',
                   '|-damage|p2a: Starmie|39/100',
                   '|replace|p2a: Zoroark|Zoroark-Hisui, L50', '|-end|p2a: Zoroark|Illusion', '|turn|2'])
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=lambda sid: sid, rng=SimpleNamespace(shuffle=lambda x: None),
                                     sample=lambda sid, *a: dict(CHOMP, species=sid))
    side = engine.opponent_side(session.decision())
    real = session.battle.opponent_active_pokemon
    assert 'shadowball' in real.moves and real.last_move.id == 'shadowball'
    assert real.boosts['spa'] == 2 and real.item is None
    assert len(side['team']) == 3
    assert side['totalFainted'] == 0
    assert side['mons'][0]['hp'] == .39
    assert all(m['hp'] == 1 for m in side['mons'][1:])


def test_illusion_restores_the_imitated_members_genuine_prior_history():
    session = session_for([CHOMP])
    session.apply(['|switch|p1a: Garchomp|Garchomp, L50|200/200',
                   '|switch|p2a: Starmie|Starmie, L50|100/100', '|turn|1',
                   '|move|p2a: Starmie|Surf|p1a: Garchomp',
                   '|-damage|p2a: Starmie|90/100|[from] item: Life Orb',
                   '|switch|p2a: Lucario|Lucario, L50|100/100', '|turn|2',
                   '|switch|p2a: Starmie|Starmie, L50|100/100', '|turn|3',
                   '|move|p2a: Starmie|Bitter Malice|p1a: Garchomp',
                   '|-enditem|p2a: Starmie|Focus Sash', '|-damage|p2a: Starmie|1/100',
                   '|replace|p2a: Zoroark|Zoroark-Hisui, L50', '|turn|4'])
    real = session.battle.opponent_active_pokemon
    assert set(real.moves) == {'bittermalice'} and real.item is None
    imitated = session.battle.opponent_team['p2: Starmie']
    assert set(imitated.moves) == {'surf'}
    assert imitated.item == 'lifeorb' and imitated.current_hp_fraction == .9
    assert not imitated.active


def test_illusion_damage_cannot_constrain_the_imitated_species_posterior():
    from pokesim.belief.observe import Observer

    observer = Observer([CHOMP], lambda sid: sid)
    log = ['|poke|p2|Starmie, L50|', '|poke|p2|Zoroark-Hisui, L50|', '|teampreview|3',
           '|switch|p1a: Garchomp|Garchomp, L50|200/200',
           '|switch|p2a: Starmie|Starmie, L50|100/100', '|turn|1',
           '|-item|p2a: Starmie|Choice Scarf|[from] ability: Frisk',
           '|move|p2a: Starmie|Shadow Ball|p1a: Garchomp', '|-damage|p1a: Garchomp|150/200', '|turn|2']
    assert len(observer.feed(log)) == 1
    assert observer.observations[0].species == 'starmie'
    log += ['|replace|p2a: Zoroark|Zoroark-Hisui, L50', '|turn|3']
    observer.feed(log)
    assert not observer.observations and len(observer.quarantined) == 1
    assert observer.item_epochs == {}
    log += ['|move|p2a: Zoroark|Shadow Ball|p1a: Garchomp', '|-damage|p1a: Garchomp|100/200', '|turn|4']
    fresh = observer.feed(log)
    assert len(fresh) == 1 and fresh[0].species == 'zoroarkhisui'
