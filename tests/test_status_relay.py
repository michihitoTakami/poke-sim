"""Recent failures reproduced at protocol/simulator boundaries, without labeling manual moves as teachers."""
import json
from types import SimpleNamespace

import pytest
from test_expert_continuation import DIRGE, FLOETTE, GHOLDENGO, PASSER, SPEC, side
from test_proposals37 import planned_sample
from test_relay_goals import GOAL, UMBREON
from test_worker_transitions import worker

from pokesim.live.observed import collect, load_records, same_record
from pokesim.live.session import BattleSession
from pokesim.search.lookahead import Lookahead, mon_state

BELL = dict(DIRGE, species='Bellibolt', ability='Electromorphosis', item='',
            moves=['Toxic', 'Soak', 'Tackle', 'Slack Off'])
PRIM = dict(GHOLDENGO, species='Primarina', ability='Torrent', item='Leftovers', evs=SPEC,
            moves=['Moonblast', 'Sparkling Aria', 'Calm Mind', 'Substitute'])


def test_soak_crosses_the_live_reconstruction_boundary_and_resets_on_a_normal_switch():
    session = BattleSession([DIRGE, GHOLDENGO])
    session.preview(['Bellibolt'])
    session.select([0, 1])
    name = session.info[0]['name']
    session.apply([f'|switch|p1a: {name}|Skeledirge, L50|211/211',
                   '|switch|p2a: Bell|Bellibolt, L50|100/100', '|turn|1',
                   f'|-start|p1a: {name}|typechange|Water', '|turn|2'])
    b = session.decision()
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=lambda s: s)
    ours = engine.own_side(b, {'skeledirge': DIRGE, 'gholdengo': GHOLDENGO})
    assert ours['mons'][0]['types'] == ['Water']
    sample = {'p1': ours, 'p2': side([dict(BELL, moves=['Thunderbolt'])]), 'field': {}}
    result = worker("""
const {expectedDamage}=sandbox.module.exports;
const b=setUp(sample,format,1), before=b.p1.active[0].getTypes();
step(b,'switch 2','move 1');step(b,'switch 2','move 1');
console.log(JSON.stringify({before,after:b.p1.active[0].getTypes()}));b.destroy();
""", sample)
    assert result == {'before': ['Water'], 'after': ['Fire', 'Ghost']}
    mon = b.active_pokemon
    mon.switch_out({})
    assert 'types' not in mon_state(mon, False, True)


def test_soak_changes_electric_effectiveness_and_removes_fire_stab_in_public_damage():
    sample = {'p1': side([DIRGE]), 'p2': side([dict(BELL, moves=['Thunderbolt'])]), 'field': {}}
    result = planned_sample(sample, """
const {estimateDamage}=require('./services/sim-tools/public_damage');
const view=obs.view(b.p1), me=view.request.side.pokemon.find(m=>m.active), foe=view.public.mons.p2.bellibolt;
const hit=()=>estimateDamage(view,b.dex,foe,me,b.dex.moves.get('thunderbolt'));
const fire=()=>estimateDamage(view,b.dex,me,foe,b.dex.moves.get('torchsong'));
const before=[hit(),fire()];obs.append(['|-start|p1a: '+b.p1.active[0].name+'|typechange|Water']);
const afterView=obs.view(b.p1);view.public=afterView.public;
console.log(JSON.stringify({before,after:[hit(),fire()],types:view.public.mons.p1.skeledirge.types}));b.destroy();
""")
    assert result['types'] == ['Water']
    assert result['after'][0] == pytest.approx(2 * result['before'][0])
    assert result['after'][1] == pytest.approx(result['before'][1] / 1.5)


def test_soak_resets_on_mega_evolution_in_both_protocol_and_reconstructed_engine():
    mon = dict(FLOETTE, item='Floettite')
    session = BattleSession([mon])
    session.preview(['Bellibolt'])
    session.select([0])
    name = session.info[0]['name']
    session.apply([f'|switch|p1a: {name}|Floette-Eternal, L50|173/173',
                   '|switch|p2a: Bell|Bellibolt, L50|100/100', '|turn|1',
                   f'|-start|p1a: {name}|typechange|Water', '|turn|2'])
    assert mon_state(session.battle.active_pokemon, True)['types'] == ['Water']
    session.apply([f'|detailschange|p1a: {name}|Floette-Mega, L50',
                   f'|-mega|p1a: {name}|Floette-Eternal|Floettite', '|turn|3'])
    state = mon_state(session.battle.active_pokemon, True)
    assert 'types' not in state
    assert [t.name for t in session.battle.active_pokemon.types] == ['FAIRY']
    result = worker("""
const b=setUp(sample,format,1);step(b,'move 1 mega','move 3');
console.log(JSON.stringify(b.p1.active[0].getTypes()));b.destroy();
""", {'p1': side([mon], [{'hp': 1, 'types': ['Water']}]), 'p2': side([BELL]), 'field': {}})
    assert result == ['Fairy']


@pytest.mark.parametrize('mid', ['Toxic', 'Soak', 'Yawn'])
def test_substitute_prevents_revealed_status_before_it_lands(mid):
    sample = {'p1': side([FLOETTE]), 'p2': side([dict(BELL, moves=[mid, 'Tackle', 'Slack Off', 'Protect'])]),
              'field': {}}
    result = planned_sample(sample, GOAL + """
const s=relayStrategy(obs.view(b.p1),b.dex),a=next();advance(a[0]);
console.log(JSON.stringify({goal:s?.goal,choice:a[1],status:b.p1.active[0].status,
 types:b.p1.active[0].getTypes(),yawn:!!b.p1.active[0].volatiles.yawn}));b.destroy();
""")
    assert result['goal'] == 'prevent-status-with-substitute'
    assert result['choice'] == 'move:substitute' and result['status'] == ''
    assert result['types'] == ['Fairy'] and not result['yawn']


@pytest.mark.parametrize('mid', ['Toxic', 'Soak'])
def test_status_immune_switch_uses_good_as_gold_against_both_moves(mid):
    sample = {'p1': side([DIRGE, GHOLDENGO]), 'p2': side([dict(BELL, moves=[mid, 'Tackle', 'Slack Off', 'Protect'])]),
              'field': {}}
    result = planned_sample(sample, GOAL + """
const s=relayStrategy(obs.view(b.p1),b.dex),a=next();advance(a[0]);
console.log(JSON.stringify({goal:s?.goal,choice:a[1],status:b.p1.active[0].status,
 types:b.p1.active[0].getTypes()}));b.destroy();
""")
    assert result['goal'] == 'switch-to-status-immunity' and result['choice'] == 'switch:gholdengo'
    assert result['status'] == '' and result['types'] == ['Steel', 'Ghost']


def test_existing_yawn_is_not_treated_as_curable_by_a_new_substitute():
    sample = {'p1': side([FLOETTE], [{'hp': 1, 'volatiles': ['yawn']}]),
              'p2': side([BELL]), 'field': {}}
    result = planned_sample(sample, GOAL + """
console.log(JSON.stringify(relayStrategy(obs.view(b.p1),b.dex)));b.destroy();
""")
    assert result is None


def test_special_defence_baton_does_not_require_a_previous_speed_boost():
    sample = {'p1': side([UMBREON, GHOLDENGO], [{'hp': 1, 'boosts': {'spa': 1, 'spd': 1}}, {'hp': 1}]),
              'p2': side([PRIM]), 'field': {}}
    result = planned_sample(sample, GOAL + """
const s=relayStrategy(obs.view(b.p1),b.dex),a=next();advance(a[0]);advance(next()[0],'pass');
console.log(JSON.stringify({goal:s?.goal,choice:a[1],receiver:b.p1.active[0].species.id,
 spd:b.p1.active[0].boosts.spd,hp:b.p1.active[0].hp}));b.destroy();
""")
    assert result['goal'] == 'special-receiver-ready' and result['choice'] == 'move:batonpass'
    assert result['receiver'] == 'gholdengo' and result['spd'] == 1 and result['hp'] > 0


def test_counterplay_punishes_passive_speed_passers_with_special_setup():
    sample = {'p1': side([PRIM]), 'p2': side([PASSER]), 'field': {}}
    result = planned_sample(sample, """
const {chooseCounterplay}=require('./services/sim-tools/counterplay_policy');
const a=chooseCounterplay(obs.view(b.p1),b.dex);advance(a[0]);
console.log(JSON.stringify({choice:a[1],spa:b.p1.active[0].boosts.spa}));b.destroy();
""")
    assert result == {'choice': 'move:calmmind', 'spa': 1}


def test_counterplay_uses_soak_to_remove_poison_immunity_then_toxic():
    sample = {'p1': side([BELL]), 'p2': side([dict(GHOLDENGO, ability='Illuminate')]), 'field': {}}
    result = planned_sample(sample, """
const {chooseCounterplay}=require('./services/sim-tools/counterplay_policy');
obs.append(['|-ability|p2a: '+b.p2.active[0].name+'|Illuminate']);
let a=chooseCounterplay(obs.view(b.p1),b.dex);advance(a[0],'move 3');
const second=chooseCounterplay(obs.view(b.p1),b.dex);advance(second[0],'move 3');
console.log(JSON.stringify({labels:[a[1],second[1]],status:b.p2.active[0].status}));b.destroy();
""")
    assert result['labels'] == ['move:soak', 'move:toxic'] and result['status'] == 'tox'


def test_learning_counts_distinct_ladder_rematches_and_replaces_illusion_identity(tmp_path):
    log = ['|switch|p2a: Cat|Meowscarada, L50|100/100', '|move|p2a: Cat|Flamethrower|p1a: Me',
           '|replace|p2a: Fox|Zoroark-Hisui, L50|100/100', '|move|p2a: Fox|Focus Blast|p1a: Me',
           '|switch|p2a: Cat|Meowscarada, L50|100/100', '|move|p2a: Cat|Knock Off|p1a: Me']
    preview = {'species': ['Meowscarada', 'Zoroark-Hisui', 'Garchomp', 'Corviknight', 'Bellibolt', 'Primarina']}
    for i in range(2):
        (tmp_path / f'{i}.json').write_text(json.dumps(dict(source=f'showdown battle-{i}',
            started=f'20261020-010{i}00', preview=preview, log=log)))
    records = load_records([tmp_path, tmp_path])  # stable identity also removes actual duplicates
    assert len(records) == 2
    obs = collect(records)
    assert obs['zoroarkhisui']['leads'] == 2 and obs['meowscarada']['leads'] == 0
    assert obs['zoroarkhisui']['moves'] == {'flamethrower': 2, 'focusblast': 2}
    assert obs['meowscarada']['moves'] == {'knockoff': 2}


def test_generic_frame_source_name_is_not_a_battle_identity():
    a = dict(source='replayed frames', started='20261020-010000',
             preview={'species': ['Primarina', 'Garchomp', 'Bellibolt', 'Corviknight']})
    assert not same_record(a, dict(a, started='20261020-020000'))
    assert same_record(a, dict(a, source='live frames', started='20261020-010100'))
    assert not same_record(a, dict(a, source='showdown battle-123'))
