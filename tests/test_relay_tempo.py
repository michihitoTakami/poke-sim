"""Observed ladder failures: alternating protection, dynamic damage, and impossible shields."""
import pytest
from test_expert_continuation import CHOMP, DIRGE, GHOLDENGO, PASSER, side
from test_proposals37 import planned_sample
from test_worker_transitions import MENCE, worker

REGIELEKI = {"species": "Regieleki", "ability": "Transistor", "nature": "Serious", "item": "",
             "moves": ["Thunderbolt"], "level": 50, "evs": {"hp": 32, "spa": 32, "spe": 0}}


def test_unknown_first_attack_is_scouted_before_committing_to_iron_defense():
    sample = {'p1': side([PASSER, DIRGE]), 'p2': side([CHOMP]), 'field': {}}
    result = planned_sample(sample, """
const {chooseBatonPublic}=require('./services/sim-tools/public_policy');
const view=obs.view(b.p1);view.public.mons.p2[view.public.active.p2].moves=[];
view.public.mons.p1[view.public.active.p1].lastMove=null;
console.log(JSON.stringify(chooseBatonPublic(view,'expert',b.dex)[1]));b.destroy();
""")
    assert result == 'move:protect'


@pytest.mark.parametrize("controller", ["expert", "planned"])
def test_protect_substitute_alternates_until_the_slow_receiver_outspeeds(controller):
    sample = {'p1': side([PASSER, DIRGE]), 'p2': side([REGIELEKI]), 'field': {}}
    result = planned_sample(sample, """
const {chooseBatonPublic}=require('./services/sim-tools/public_policy');
const labels=[],hp=[];
for(let i=0;i<5;i++) {
 const view=obs.view(b.p1);
 const action=CONTROLLER==='planned'?p.choose(view,b.dex):chooseBatonPublic(view,'expert',b.dex);
 labels.push(action[1]);hp.push(b.p1.active[0].hp);
 if(action[1]==='move:batonpass')break;
 advance(action[0]);
}
console.log(JSON.stringify({labels,hp,speed:b.p1.active[0].boosts.spe}));b.destroy();
""".replace('CONTROLLER', repr(controller)))
    assert result['labels'] == ['move:protect', 'move:substitute', 'move:protect', 'move:substitute',
                                'move:batonpass']
    assert result['hp'] == [165, 165, 124, 124, 83]
    assert result['speed'] == 4


def test_replanning_the_live_root_each_turn_preserves_the_protection_sequence():
    sample = {'p1': side([PASSER, DIRGE]), 'p2': side([REGIELEKI]), 'field': {}}
    result = planned_sample(sample, """
const {Search}=require('./services/sim-tools/terminal_search');
const {State}=require(process.argv[2]+'/dist/sim/state');
const labels=[];
for(let i=0;i<5;i++) {
 const current=JSON.parse(JSON.stringify(sample));
 for(const side of ['p1','p2']) current[side].mons=b[side].pokemon.map(m=>({hp:m.hp/m.maxhp,
   boosts:{...m.boosts},lastMove:m.lastMove?.id,
   stall:m.volatiles.stall?Math.log(m.volatiles.stall.counter)/Math.log(3):0}));
 const search=new Search({format,samples:[current],observationLog:obs.history},
   {setUp,step,choices,State,extractChannelMessages});
 const label=search.reply().results[0].rootPlan;labels.push(label);
 if(label==='move:batonpass')break;
 advance(choices(b,b.p1).find(([,l])=>l===label)[0]);
}
console.log(JSON.stringify(labels));b.destroy();
""")
    assert result == ['move:protect', 'move:substitute', 'move:protect', 'move:substitute', 'move:batonpass']


def test_ready_receiver_passes_instead_of_starting_another_protection_cycle():
    sample = {'p1': side([PASSER, DIRGE], [{'hp': .51, 'boosts': {'spe': 4}}, {'hp': 1}]),
              'p2': side([REGIELEKI]), 'field': {}}
    result = planned_sample(sample, "console.log(JSON.stringify(next()[1]));b.destroy();")
    assert result == 'move:batonpass'


def test_protection_can_wait_out_sun_when_the_fast_receiver_cannot_survive_entry():
    foe = dict(REGIELEKI, species='Charizard', ability='Drought', moves=['Weather Ball'])
    sample = {'p1': side([PASSER, GHOLDENGO], [{'hp': .75, 'boosts': {'spe': 2}}, {'hp': 1}]),
              'p2': side([foe]), 'field': {'weather': 'sunnyday'}}
    result = planned_sample(sample, """
obs.append(['|-weather|SunnyDay']);const waiting=next()[1];
obs.append(['|-weather|none']);const pass=next()[1];
console.log(JSON.stringify({waiting,pass}));b.destroy();
""")
    assert result == {'waiting': 'move:protect', 'pass': 'move:batonpass'}


def test_priority_that_would_kill_after_substitute_triggers_handoff_instead():
    foe = dict(REGIELEKI, species='Basculegion', ability='Adaptability', moves=['Surf', 'Aqua Jet'])
    sample = {'p1': side([PASSER, DIRGE], [{'hp': .4, 'boosts': {'spe': 1}, 'lastMove': 'protect'}, {'hp': 1}]),
              'p2': side([foe]), 'field': {}}
    result = planned_sample(sample, """
obs.append(['|move|p1a: '+b.p1.active[0].name+'|Protect|p1a: '+b.p1.active[0].name,
 '|-ability|p2a: '+b.p2.active[0].name+'|Adaptability']);
console.log(JSON.stringify(next()[1]));b.destroy();
""")
    assert result != 'move:substitute'


def test_substitute_is_not_used_when_sand_would_ko_before_the_next_handoff():
    sample = {'p1': side([PASSER, DIRGE], [{'hp': 42 / 165, 'boosts': {'spe': 1}}, {'hp': 1}]),
              'p2': side([REGIELEKI]), 'field': {'weather': 'sandstorm'}}
    result = planned_sample(sample, """
obs.append(['|-weather|Sandstorm','|move|p1a: '+b.p1.active[0].name+'|Protect|p1a: '+b.p1.active[0].name]);
console.log(JSON.stringify(next()[1]));b.destroy();
""")
    assert result == 'move:batonpass'


def damage_probe(sample, script):
    return planned_sample(sample, """
const {estimateDamage,megaSource}=require('./services/sim-tools/public_damage');
const view=obs.view(b.p1),me=view.request.side.pokemon.find(m=>m.active);
const foe=view.public.mons.p2[view.public.active.p2];
const damage=(source,target,mid,a,c)=>estimateDamage(view,b.dex,source,target,b.dex.moves.get(mid),a,c);
""" + script)


def test_sun_weather_ball_is_a_strong_fire_attack_and_cloud_nine_cancels_it():
    foe = dict(CHOMP, species='Charizard', ability='Drought', item='', moves=['Weather Ball'])
    result = damage_probe({'p1': side([PASSER]), 'p2': side([foe]), 'field': {'weather': 'sunnyday'}}, """
view.public.weather='SunnyDay';
const sun=damage(foe,me,'weatherball');view.public.weather='';const plain=damage(foe,me,'weatherball');
view.public.weather='SunnyDay';me.ability='cloudnine';const suppressed=damage(foe,me,'weatherball');
console.log(JSON.stringify({sun,plain,suppressed}));b.destroy();
""")
    assert result['sun'] > 6 * result['plain']
    assert result['suppressed'] == pytest.approx(result['plain'])


def test_aerilate_and_the_known_mega_stone_make_double_edge_hit_ghosts():
    result = damage_probe({'p1': side([MENCE]), 'p2': side([GHOLDENGO]), 'field': {}}, """
const before=damage(me,foe,'doubleedge'),after=damage(megaSource(view,b.dex,me),foe,'doubleedge');
console.log(JSON.stringify({before,after}));b.destroy();
""")
    assert result['before'] == 0
    assert result['after'] > 0


def test_fairy_aura_is_applied_when_the_known_stone_is_about_to_activate_it():
    from test_expert_continuation import FLOETTE

    ours = dict(FLOETTE, item='Floettite')
    result = damage_probe({'p1': side([ours]), 'p2': side([CHOMP]), 'field': {}}, """
const mega=megaSource(view,b.dex,me),withAura=damage(mega,foe,'moonblast');
const withoutAura=damage({...mega,ability:'flowerveil'},foe,'moonblast');
console.log(JSON.stringify({withAura,withoutAura}));b.destroy();
""")
    assert result['withAura'] == pytest.approx(result['withoutAura'] * 4 / 3)


def test_body_press_uses_defence_boosts_and_not_attack_boosts():
    foe = dict(CHOMP, species='Aggron', ability='Sturdy', item='', moves=['Body Press'])
    result = damage_probe({'p1': side([MENCE]), 'p2': side([foe]), 'field': {}}, """
const plain=damage(foe,me,'bodypress',{}),defence=damage(foe,me,'bodypress',{def:4}),
 attack=damage(foe,me,'bodypress',{atk:6});
console.log(JSON.stringify({plain,defence,attack}));b.destroy();
""")
    assert result['defence'] > 2 * result['plain']
    assert result['attack'] == pytest.approx(result['plain'])


def test_roost_cannot_rely_on_flying_resistance_after_the_faster_heal():
    foe = dict(CHOMP, species='Aggron', ability='Sturdy', item='', moves=['Body Press'])
    sample = {'p1': side([MENCE], [{'hp': 47 / 202, 'mega': True}]),
              'p2': side([foe], [{'hp': .5, 'boosts': {'def': 6}}]), 'field': {}}
    result = damage_probe(sample, """
const {chooseBatonPublic}=require('./services/sim-tools/public_policy');
const before=damage(foe,me,'bodypress'),after=damage(foe,{...me,types:['Dragon']},'bodypress');
console.log(JSON.stringify({before,after,choice:chooseBatonPublic(view,'tactical',b.dex)[1]}));b.destroy();
""")
    assert result['after'] == pytest.approx(2 * result['before'])
    assert result['choice'] != 'move:roost'


def test_last_respects_uses_cumulative_public_faints_including_revivals():
    foe = dict(CHOMP, species='Basculegion', ability='Adaptability', item='', moves=['Last Respects'])
    result = damage_probe({'p1': side([DIRGE]), 'p2': side([foe]), 'field': {}}, """
const plain=damage(foe,me,'lastrespects');
const name=b.p2.active[0].name;
obs.append(['|faint|p2a: '+name,'|-heal|p2a: '+name+'|50/100|[from] move: Revival Blessing',
 '|faint|p2a: '+name,'|-heal|p2a: '+name+'|50/100|[from] move: Revival Blessing']);
view.public=obs.view(b.p1).public;
const raised=damage(foe,me,'lastrespects');
console.log(JSON.stringify({plain,raised,faints:view.public.totalFainted.p2}));b.destroy();
""")
    assert result['faints'] == 2
    assert result['raised'] > 2.5 * result['plain']


@pytest.mark.parametrize('foe_move,excluded', [('Weather Ball', True), ('Heal Pulse', False)])
def test_root_excludes_hp_failure_only_when_no_preceding_reply_can_restore_hp(foe_move, excluded):
    foe = dict(REGIELEKI, moves=[foe_move])
    sample = {'p1': side([PASSER, GHOLDENGO], [{'hp': 1 / 165}, {'hp': 1}]),
              'p2': side([foe]), 'field': {}}
    result = worker("""
const {State}=require(process.argv[2]+'/dist/sim/state');
const {pruneRoot}=require('./services/sim-tools/action_filter');
const b=setUp(sample,format,1),st={snapshot:JSON.stringify(State.serializeBattle(b)),
 p1:choices(b,b.p1),p2:choices(b,b.p2)};
const info=pruneRoot([st],{State});console.log(JSON.stringify(info));b.destroy();
""", sample)
    assert any(e['label'] == 'move:substitute' for e in result['excluded']) == excluded


def test_live_root_tie_uses_the_relay_plan_before_lucky_deep_win_speed():
    from pokesim.search.planner import plan

    row = {'p1': ['move:protect', 'move:substitute'], 'p2': ['move:weatherball'],
           'payoff': [[-.609], [-.638]], 'winSpeed': [[.01], [.1]], 'visits': [[20], [20]],
           'rootPlan': 'move:protect',
           'terminal_search': {'immediate_wins': []}}
    advice = plan([row], pure=True)
    assert advice.ranked()[0][0] == 'move:protect'
    assert advice.status['relay_plan']['followed']
    row['payoff'][1][0] = -.4
    assert plan([row], pure=True).ranked()[0][0] == 'move:substitute'
    row['payoff'][1][0] = -.638
    incompatible = dict(row, rootPlan='move:substitute')
    assert plan([row, incompatible], pure=True).ranked()[0][0] == 'move:substitute'
    row['terminal_search']['immediate_wins'] = ['move:substitute']
    assert plan([row], pure=True).ranked()[0][0] == 'move:substitute'


def test_passer_led_trios_get_full_games_even_when_static_coverage_ranks_them_last():
    import itertools
    import random
    from types import SimpleNamespace

    from pokesim.live.lineup_search import LineupSearch
    from pokesim.live.selection import OpponentPicks

    sets = [{'species': name, 'moves': ['Baton Pass'] if name in {'scolipede', 'umbreon'} else ['Tackle']}
            for name in ['scolipede', 'umbreon', 'a', 'b', 'c', 'd']]
    names = [s['species'] for s in sets]
    lineups = [[lead, *sorted(set(trio) - {lead})] for trio in itertools.combinations(names, 3) for lead in trio]
    lineups.sort(key=lambda trio: trio[0] in {'scolipede', 'umbreon'})
    ranking = [(trio, 1 - i / 100, {}) for i, trio in enumerate(lineups)]
    desired = ['scolipede', 'a', 'umbreon']
    calls = []

    def query(req):
        cfg = req['lineupMatches']
        calls.append(cfg)
        return {'version': 'test', 'informationScope': 'public_observation', 'trials': [
            {'candidate': i, 'world': w, 'opponent': policy, 'target': float(trio == desired),
             'outcome': 'win' if trio == desired else 'loss'}
            for i, trio in enumerate([[s['species'] for s in team] for team in cfg['ours']])
            for w in range(len(cfg['worlds'])) for policy in cfg['policies']]}
    engine = SimpleNamespace(sampler=SimpleNamespace(rng=random.Random(0), sample=lambda s: {'species': s}),
                             _query=query)
    picks = OpponentPicks(['x', 'y', 'z'], [1., 1., 1.], [1., 0., 0.], [(['x', 'y', 'z'], 1.)])
    ranked, info = LineupSearch(sets, engine).rerank(ranking, picks)
    tested = [[s['species'] for s in team] for team in calls[-1]['ours']]
    assert sum(trio[0] in {'scolipede', 'umbreon'} for trio in tested) == 20
    assert len(tested) == len({tuple(t) for t in tested})
    assert ranked[0][0] == desired
    assert len({tuple(sorted(team)) for team in tested}) == 20
    assert info['calibrated_win_probability'] is False


@pytest.mark.parametrize('controller', ['expert', 'planned'])
def test_completed_payload_is_not_passed_back_to_the_passive_passer(controller):
    umbreon = dict(DIRGE, species='Umbreon', ability='Inner Focus', item='Leftovers',
                   moves=['Snarl', 'Baton Pass', 'Calm Mind', 'Moonlight'])
    sample = {'p1': side([umbreon, PASSER, GHOLDENGO],
                        [{'hp': 1, 'boosts': {'def': 4, 'spe': 4, 'spa': 2, 'spd': 2}}, {'hp': 1}, {'hp': 0}]),
              'p2': side([dict(CHOMP, moves=['Dragon Claw'], item='')]), 'field': {}}
    result = planned_sample(sample, """
const {chooseBatonPublic}=require('./services/sim-tools/public_policy');
const labels=[];
for(let i=0;i<12;i++) {
 const view=obs.view(b.p1);
 const choice=CONTROLLER==='planned'?p.choose(view,b.dex):chooseBatonPublic(view,'expert',b.dex);
 labels.push(choice[1]);
 if(choice[1]==='move:snarl'||choice[1]==='move:batonpass')break;
 advance(choice[0]);
}
console.log(JSON.stringify(labels));b.destroy();
""".replace('CONTROLLER', repr(controller)))
    assert 'move:batonpass' not in result and result[-1] == 'move:snarl'


def test_a_safe_passive_relay_can_still_add_missing_defence_for_a_living_attacker():
    umbreon = dict(DIRGE, species='Umbreon', ability='Inner Focus', item='Leftovers',
                   moves=['Snarl', 'Baton Pass', 'Calm Mind', 'Moonlight'])
    weak = dict(CHOMP, species='Rattata', ability='Run Away', moves=['Tackle'], item='')
    sample = {'p1': side([umbreon, PASSER], [{'hp': 1, 'boosts': {'spa': 2, 'spd': 2}}, {'hp': 1}]),
              'p2': side([weak]), 'field': {}}
    result = planned_sample(sample, "console.log(JSON.stringify(next()[1]));b.destroy();")
    assert result == 'move:batonpass'
