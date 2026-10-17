"""Mechanism controls for #37: public timers, action provenance, plans, finite speed worlds and team games."""
import copy
from types import SimpleNamespace

import pytest
from test_expert_continuation import CHOMP, FLOETTE, HIPPO, PASSER, side
from test_worker_transitions import worker

from pokesim.live.provenance import actual_lineup, execution_rows, reconstruction_lineup
from pokesim.live.session import BattleSession
from pokesim.search.lookahead import Lookahead, to_id
from pokesim.search.planner import plan


def test_yawn_age_is_exported_and_sleeps_at_the_next_real_residual():
    sets = [dict(FLOETTE, moves=['Substitute', 'Calm Mind', 'Moonblast', 'Synthesis'])]
    session = BattleSession(sets)
    session.preview(['Hippowdon'])
    session.select([0])
    name = session.info[0]['name']
    session.apply([f'|switch|p1a: {name}|Floette-Eternal, L50|150/173',
                   '|switch|p2a: Hippo|Hippowdon, L50|100/100', '|turn|4',
                   f'|-start|p1a: {name}|move: Yawn', '|upkeep', '|turn|5'])
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=lambda s: s)
    ours = engine.own_side(session.decision(), {to_id(sets[0]['species']): sets[0]})
    assert ours['mons'][0]['volatileTurns'] == {'yawn': 1}
    sample = {'p1': ours, 'p2': side([dict(HIPPO, moves=['Splash'])]), 'field': {}}
    result = worker("""
const b=setUp(sample,format,1);step(b,'move 1','move 1');
console.log(JSON.stringify({status:b.p1.active[0].status,sub:!!b.p1.active[0].volatiles.substitute}));b.destroy();
""", sample)
    # Putting up a Substitute after becoming drowsy does not remove the existing Yawn.
    assert result == {'status': 'slp', 'sub': True}


def test_public_fallen_counter_survives_revival_and_can_override_current_entry_multiplier():
    from pokesim.state.tracker import ProtocolTracker

    tracker = ProtocolTracker()
    for line in ['|switch|p2a: King|Kingambit, L50|100/100', '|-start|p2a: King|fallen3|[silent]']:
        tracker.apply(line.split('|'))
    assert tracker.total_fainted['p2'] == 3
    assert tracker.locks('p2', 'King')['fallen'] == 3
    sample = {'p1': side([dict(CHOMP, species='Kingambit', ability='Supreme Overlord')],
                         [{'fallen': 3}]), 'p2': side([HIPPO]), 'field': {}}
    sample['p1']['totalFainted'] = 4
    result = worker("""
const b=setUp(sample,format,1);console.log(JSON.stringify({n:b.p1.totalFainted,
 fallen:b.p1.active[0].abilityState.fallen}));b.destroy();
""", sample)
    assert result == {'n': 4, 'fallen': 3}


def test_recover_actual_three_without_turning_a_matching_recommendation_into_ai_execution():
    rec = {'lineup': ['Gholdengo', 'Floette-Eternal', 'Skeledirge'],
           'advice': [{'turn': 1, 'chosen': 'move:protect'}],
           'log': ['|switch|p1a: S|Scolipede, L50|100/100', '|turn|1', '|move|p1a: S|Protect|p1a: S',
                   '|switch|p1a: M|Salamence, L50|100/100', '|switch|p1a: U|Umbreon, L50|100/100']}
    assert actual_lineup(rec)['species'] == ['Scolipede', 'Salamence', 'Umbreon']
    assert reconstruction_lineup(rec)[0] == ['Scolipede', 'Salamence', 'Umbreon']
    assert execution_rows(rec)[0]['source'] == 'unknown'
    rec['advice'][0]['submission'] = {'source': 'ai', 'action': 'move:protect'}
    assert execution_rows(rec)[0]['source'] == 'unknown'
    rec['control_mode'] = 'autonomous_ai'
    assert execution_rows(rec)[0]['source'] == 'ai'
    rec['execution_sources'] = [{'turn': 1, 'action': 'move:protect', 'source': 'manual'}]
    assert execution_rows(rec)[0]['source'] == 'manual'


def test_partial_mismatched_lineup_does_not_fill_unseen_bench_from_future_switches():
    rec = {'lineup': ['Gholdengo', 'Floette-Eternal', 'Skeledirge'],
           'log': ['|switch|p1a: S|Scolipede, L50|100/100', '|turn|1',
                   '|turn|2', '|switch|p1a: M|Salamence, L50|100/100', '|turn|3']}
    names, info = reconstruction_lineup(rec, 1)
    assert names == ['Scolipede'] and not info['complete']
    assert all('Gholdengo' not in n for n in names)


def test_uncertainty_cannot_enable_the_old_point_one_hp_tiebreak():
    result = {'p1': ['move:moonblast', 'switch:umbreon'], 'p2': ['move:expandingforce'],
              'payoff': [[.239], [.289]], 'winSpeed': [[.1], [0]], 'rootProgress': [[1.], [0.]],
              'uncertainty': [[.4], [.4]], 'visits': [[2], [2]],
              'terminal_search': {'immediate_wins': [], 'value_kind': 'test'}}
    assert plan([result], pure=True).ranked()[0][0] == 'switch:umbreon'
    assert plan([result], pure=True).status['tie_margin'] == .03
    assert plan([result], pure=True, tie_margin=.1).ranked()[0][0] == 'move:moonblast'


def planned_sample(sample, script):
    return worker("""
const {PublicObservation}=require('./services/sim-tools/public_observation');
const {PlanPolicy,assess}=require('./services/sim-tools/public_plans');
const {extractChannelMessages}=require(process.argv[2]+'/dist/sim/battle');
const b=setUp(sample,format,1), log=[];
for(const side of b.sides) {
 const m=side.active[0], who=side.id+'a: '+m.name;
 log.push(`|switch|${who}|${m.species.name}, L50|${m.hp}/${m.maxhp}`);
 for(const [s,n] of Object.entries(m.boosts)) if(n) log.push(`|-setboost|${who}|${s}|${n}`);
 for(const v of Object.keys(m.volatiles)) log.push(`|-start|${who}|${v}`);
 if(side.id==='p2') for(const s of m.moveSlots) log.push(`|move|${who}|${s.move}|${who}`);
}
log.push('|turn|2');const obs=new PublicObservation(log,b.dex,extractChannelMessages), p=new PlanPolicy();
let pos=b.log.length;
function next(){const answer=p.choose(obs.view(b.p1),b.dex);return answer;}
function advance(choice,foe='move 1'){step(b,choice,foe);obs.append(b.log.slice(pos),true);pos=b.log.length;}
""" + script, sample)


def test_substitute_setup_attack_plan_has_a_goal_and_interrupts_on_new_public_move():
    sample = {'p1': side([dict(FLOETTE, item='', moves=['Substitute', 'Calm Mind', 'Moonblast'])]),
              'p2': side([dict(HIPPO, item='', moves=['Slack Off', 'Yawn', 'Splash', 'Protect'])]), 'field': {}}
    result = planned_sample(sample, """
const labels=[];labels.push(next()[1]);advance(next()[0]);labels.push(next()[1]);
advance(next()[0]);labels.push(next()[1]);
const me=b.p1.active[0],foe=b.p2.active[0];
p.plan={kind:'protected-setup',signature:assess(obs.view(b.p1),b.dex).signature,start:obs.view(b.p1).turn};
obs.append([`|move|p2a: ${foe.name}|Roar|p1a: ${me.name}`],true);next();
console.log(JSON.stringify({labels,events:p.events}));b.destroy();
""")
    assert result['labels'] == ['move:substitute', 'move:calmmind', 'move:moonblast']
    assert any(e['event'] == 'new-public-information' for e in result['events'])


@pytest.mark.parametrize('drowsy,breaks', [(True, False), (False, True)])
def test_protection_plan_does_not_start_after_yawn_or_when_the_substitute_will_break(drowsy, breaks):
    sample = {'p1': side([dict(FLOETTE, item='', moves=['Substitute', 'Calm Mind', 'Moonblast'])],
                         [{'hp': 1, 'volatiles': ['yawn'] if drowsy else []}]),
              'p2': side([dict(HIPPO, item='', moves=['Earthquake', 'Yawn'] if breaks else ['Yawn', 'Splash'])]),
              'field': {}}
    result = planned_sample(sample, "console.log(JSON.stringify({choice:next()[1],plan:p.plan}));b.destroy();")
    assert result['plan'] is None


def test_focus_energy_revealed_interrupts_special_defence_plan_and_bypasses_positive_spdef_estimate():
    sample = {'p1': side([dict(FLOETTE, item='', moves=['Calm Mind', 'Moonblast', 'Synthesis'])],
                         [{'hp': 1, 'boosts': {'spd': 4}}]),
              'p2': side([dict(CHOMP, species='Archaludon', ability='Stamina', item='',
                               moves=['Draco Meteor', 'Focus Energy'])], [{'volatiles': ['focusenergy']}]), 'field': {}}
    result = planned_sample(sample, """
const view=obs.view(b.p1);const a=assess(view,b.dex);
view.public.mons.p1[view.public.active.p1].boosts.spd=0;const c=assess(view,b.dex);
console.log(JSON.stringify({boosted:a.incoming,neutral:c.incoming,choice:next()[1]}));b.destroy();
""")
    assert result['boosted'] == pytest.approx(result['neutral'])
    assert result['choice'] != 'move:calmmind'


def test_relay_hands_off_at_the_deadline_before_priority_kills_the_passer():
    receiver = dict(CHOMP, species='Salamence', item='Salamencite', ability='Intimidate')
    foe = dict(CHOMP, species='Basculegion', item='', ability='Adaptability', moves=['Aqua Jet', 'Ice Beam'])
    sample = {'p1': side([PASSER, receiver], [{'hp': .52, 'boosts': {'def': 2, 'spe': 1}}, {'hp': 1}]),
              'p2': side([foe]), 'field': {}}
    result = planned_sample(sample, "console.log(JSON.stringify({choice:next()[1],events:p.events}));b.destroy();")
    assert result['choice'] == 'move:batonpass'


def test_real_three_on_three_games_share_worlds_and_keep_unselected_foes_private():
    engine = Lookahead(None)
    ours = [dict(CHOMP, item='', moves=['Earthquake']), dict(FLOETTE, item='', moves=['Moonblast']),
            dict(HIPPO, item='', moves=['Earthquake'])]
    try:
        reply = engine._query({'format': 'gen9championsbssregmc', 'lineupMatches': {
            'ours': [ours, list(reversed(ours))], 'worlds': [copy.deepcopy(ours)],
            'preview': {'p1': [s['species'] for s in ours] + ['Pikachu', 'Umbreon', 'Scolipede'],
                        'p2': [s['species'] for s in ours] + ['Gholdengo', 'Alakazam', 'Starmie']},
            'policies': ['tactical'], 'maxTurns': 80}})
    finally:
        engine.close()
    assert reply['informationScope'] == 'public_observation'
    assert len(reply['trials']) == 2
    assert all(t['outcome'] in {'win', 'loss', 'draw'} and t['turns'] > 1 for t in reply['trials'])


@pytest.mark.parametrize('case,expected', [('faster', 1), ('scarf', 0), ('priority', 0), ('trickroom', 0)])
def test_action_order_is_measured_by_the_engine_including_priority_and_speed_modifiers(case, expected):
    foe = dict(HIPPO, item='', moves=['Splash'])
    if case == 'priority':
        foe['moves'] = ['Quick Attack']
    if case == 'scarf':
        foe = dict(CHOMP, item='Choice Scarf', moves=['Splash'], evs=dict(CHOMP['evs'], spe=32))
    ours = dict(CHOMP, item='', moves=['Splash'])
    sample = {'p1': side([ours]), 'p2': side([foe]), 'field': {'trickroom': 4} if case == 'trickroom' else {}}
    result = worker("""
const {State}=require(process.argv[2]+'/dist/sim/state');
const {extractChannelMessages}=require(process.argv[2]+'/dist/sim/battle');
const {Search}=require('./services/sim-tools/terminal_search');
const {observedStep}=require('./services/sim-tools/public_policy');
const history=['|switch|p1a: Garchomp|Garchomp, L50|100/100',
 '|switch|p2a: Foe|'+sample.p2.team[0].species+', L50|100/100','|turn|1'];
const s=new Search({format, samples:[sample], observationLog:history, priorVisits:0},
 {setUp,step,choices,State,extractChannelMessages,observedStep});
const pending=[];for(const _ of s.trial([0,0,0],true,pending)){}
s.feedback(pending.map(p=>({id:p.id,p:.5})));
console.log(JSON.stringify(s.reply().results[0].actionOrder[0][0]));
""", sample)
    assert result == {'first': expected, 'trials': 1}


def test_focused_search_still_covers_replies_but_only_refines_the_selected_two_rows():
    sample = {'p1': side([dict(CHOMP, moves=['Splash', 'Protect', 'Swords Dance'])]),
              'p2': side([dict(HIPPO, moves=['Splash'])]), 'field': {}}
    result = worker("""
const {State}=require(process.argv[2]+'/dist/sim/state');
const {extractChannelMessages}=require(process.argv[2]+'/dist/sim/battle');
const {Search}=require('./services/sim-tools/terminal_search');
const s=new Search({format,samples:[sample],observationLog:['|turn|1']},
 {setUp,step,choices,State,extractChannelMessages});
s.focus=[0,2];s.states[0].cells.forEach(row=>row.forEach(c=>{c.prior=0;c.n=2;}));
const rows=[];for(let k=0;k<20;k++){const job=s.pickJob();rows.push(job[1]);s.states[job[0]].cells[job[1]][job[2]].n++;}
console.log(JSON.stringify(rows));
""", sample)
    assert set(result) == {0, 2}


def test_split_guard_keeps_renamed_audited_games_in_development(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    from pokesim.learn.branches import development_battles, generate

    source = next(iter(development_battles()))
    record = tmp_path / 'renamed.json'
    record.write_text(__import__('json').dumps({'source': source}))
    positions = tmp_path / 'positions.json'
    positions.write_text('[{"file":"renamed.json","turn":1,"split":"train"}]')
    with pytest.raises(ValueError, match='renamed inspected battle'):
        generate(SimpleNamespace(positions=positions, records=tmp_path, out=tmp_path / 'out'))


def test_lineup_ranking_shares_foe_worlds_preserves_rng_and_returns_only_simulated_scores():
    import random

    from pokesim.live.lineup_search import LineupSearch
    from pokesim.live.selection import OpponentPicks

    rng = random.Random(2)
    sampler = SimpleNamespace(rng=rng, sample=lambda name: {'species': name, 'moves': ['Splash']})
    calls = []

    def query(request):
        calls.append(request)
        cfg = request['lineupMatches']
        return {'version': 'test', 'informationScope': 'public_observation', 'trials': [
            {'candidate': i, 'world': w, 'outcome': 'win' if i == 2 else 'loss', 'target': float(i == 2)}
            for i in range(len(cfg['ours'])) for w in range(len(cfg['worlds']))]}
    engine = SimpleNamespace(sampler=sampler, _query=query)
    ours = [{'species': s} for s in ['a', 'b', 'c', 'd']]
    picks = OpponentPicks(['x', 'y', 'z'], [1., 1., 1.], [1., 0., 0.], [(['x', 'y', 'z'], 1.)])
    ranking = [(['a', 'b', 'c'], .9, {}), (['b', 'a', 'c'], .8, {}),
               (['a', 'b', 'd'], .7, {}), (['a', 'c', 'd'], .6, {}), (['c', 'a', 'b'], .5, {})]
    before = rng.getstate()
    selection = LineupSearch(ours, engine)
    ranked, info = selection.rerank(ranking, picks)
    assert ranked[0][0] == ['a', 'b', 'd'] and len(ranked) == 4
    assert info['opponent_hypotheses'] == [['x', 'y', 'z']] * 4
    assert len(calls[0]['lineupMatches']['worlds']) == 4
    assert selection.rerank(ranking, picks) == (ranked, info) and len(calls) == 1
    assert sampler.rng is rng and rng.getstate() == before


def test_transform_copies_the_current_mega_stats_moves_and_five_pp_without_spending_a_mega():
    from pokesim.search.lookahead import mega_evolved

    ours = dict(FLOETTE, item='Floettite', moves=['Moonblast', 'Substitute', 'Synthesis', 'Calm Mind'])
    session = BattleSession([ours])
    session.preview(['Ditto'])
    session.select([0])
    name = session.info[0]['name']
    session.apply([f'|switch|p1a: {name}|Floette-Eternal, L50|100/173', '|turn|1',
                   f'|detailschange|p1a: {name}|Floette-Mega, L50, F',
                   f'|-mega|p1a: {name}|Floette|Floettite',
                   '|switch|p2a: Ditto|Ditto, L50|100/100',
                   f'|-transform|p2a: Ditto|p1a: {name}|[from] ability: Imposter',
                   f'|move|p2a: Ditto|Moonblast|p1a: {name}', '|turn|2'])
    battle = session.decision()
    assert mega_evolved(battle.active_pokemon) and not mega_evolved(battle.opponent_active_pokemon)
    from pokesim.search.lookahead import mon_state

    state = mon_state(battle.opponent_active_pokemon, True)
    state.update(battle.state_tracker.locks('p2', 'Ditto'))
    assert state['pp']['moonblast'] == 4
    assert state['transform']['stats']['spe'] == battle.active_pokemon.stats['spe']
    assert dict(battle.opponent_active_pokemon.base_stats) == dict(battle.active_pokemon.base_stats)
    sample = {'p1': side([ours], [{'forme': 'Floette-Mega', 'mega': True}]),
              'p2': side([dict(CHOMP, species='Ditto', ability='Imposter', item='Choice Scarf', moves=['Transform'])],
                         [state]), 'field': {}}
    result = worker("""
const b=setUp(sample,format,1),a=b.p1.active[0],c=b.p2.active[0];
console.log(JSON.stringify({forme:c.species.name,transformed:c.transformed,hp:c.maxhp,
 ourhp:a.maxhp,stats:c.storedStats,ours:a.storedStats,mega:b.p2.megaUsed,
 moves:c.moveSlots.map(m=>[m.id,m.pp,m.maxpp])}));b.destroy();
""", sample)
    assert result['forme'] == 'Floette-Mega' and result['transformed']
    assert result['stats'] == result['ours']
    assert result['hp'] != result['ourhp']  # Transform preserves Ditto's own HP stat.
    assert result['moves'][0] == ['moonblast', 4, 5]
    assert all(m[2] == 5 for m in result['moves'])
