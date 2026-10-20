"""Take a real finishing move; never certify lucky, blocked, or non-terminal KOs as a match win."""
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from test_expert_continuation import CHOMP, HIPPO, PASSER, side

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.search.lookahead import Lookahead
from pokesim.search.planner import World, plan, solve_common
from pokesim.search.terminal import search


def certificate(sample, move='move:dragonclaw', reply='move:splash', limit=4096):
    js = r"""
const fs=require('fs'),vm=require('vm');
const sandbox={__dirname:process.cwd()+'/services/sim-tools',
 require:n=>n==='readline'?{createInterface:()=>({on:()=>{}})}:require(n),
 process:{argv:['node','worker',process.argv[1]]},module:{exports:{}}};
vm.runInNewContext(fs.readFileSync('./services/sim-tools/lookahead.js','utf8')+
 '\nmodule.exports={setUp,step,choices,State};',sandbox);
const d=sandbox.module.exports, req=JSON.parse(process.argv[2]);
const b=d.setUp(req.sample,'gen9championsbssregmc',1);
const a=d.choices(b,b.p1).find(x=>x[1]===req.move)[0];
const c=d.choices(b,b.p2).find(x=>x[1]===req.reply)[0];
const {immediateWin}=require('./services/sim-tools/terminal_search');
console.log(JSON.stringify(immediateWin(JSON.stringify(d.State.serializeBattle(b)),a,c,d,Date.now()+5000,req.limit)));
b.destroy();
"""
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR),
                          json.dumps(dict(sample=sample, move=move, reply=reply, limit=limit))],
                         cwd=ROOT, text=True, capture_output=True, check=True, timeout=10)
    return json.loads(out.stdout)


def finish_sample():
    return {'p1': side([dict(CHOMP, item='', moves=['Dragon Claw', 'Swords Dance'])]),
            'p2': side([dict(HIPPO, ability='', item='', moves=['Splash'])], [{'hp': .001}]), 'field': {}}


def test_discrete_chance_exhaustion_certifies_all_damage_and_critical_rolls():
    result = certificate(finish_sample())
    assert result['win'] and result['complete'] and result['tested'] > 16


@pytest.mark.parametrize('case', ['miss', 'protect', 'bench', 'sash', 'dies_first', 'cap'])
def test_a_possible_escape_or_incomplete_enumeration_is_never_a_certificate(case):
    sample = finish_sample()
    move, reply, limit = 'move:dragonclaw', 'move:splash', 4096
    if case == 'miss':
        sample['p1']['team'][0]['moves'] = ['Dragon Rush']
        move = 'move:dragonrush'
    elif case == 'protect':
        sample['p2']['team'][0]['moves'] = ['Protect']
        reply = 'move:protect'
    elif case == 'bench':
        sample['p2']['team'].append(dict(HIPPO))
        sample['p2']['mons'].append({'hp': 1})
    elif case == 'sash':
        sample['p2']['mons'][0]['hp'] = 1
        sample['p2']['team'][0].update(item='Focus Sash', species='Shedinja', ability='Wonder Guard')
        sample['p1']['team'][0]['moves'] = ['Fire Fang']
        move = 'move:firefang'
    elif case == 'dies_first':
        sample['p1']['mons'][0]['hp'] = .001
        sample['p2']['team'][0]['moves'] = ['Extreme Speed']
        reply = 'move:extremespeed'
    else:
        limit = 1
    assert not certificate(sample, move, reply, limit)['win']


def test_fast_finish_breaks_equal_win_values_without_sacrificing_win_probability():
    world = World(np.ones((2, 2)), ['a', 'b'], predicted=np.array([.5, .5]))
    result = solve_common([world], secondary=np.array([.1, .5]))
    assert result.prob[1] == pytest.approx(1)
    world.payoff[1] = .99
    result = solve_common([world], secondary=np.array([.1, .5]))
    assert result.prob[0] > .99999
    assert result.status['objective'] > .99999


class FlatValue:
    version = {'dataset_id': 'test'}

    def probabilities(self, observations):
        return np.full(len(observations), .99)


def test_real_turn14_chooses_moonblast_even_if_every_open_leaf_looks_nearly_won():
    req = json.loads((Path(__file__).parent / 'fixtures/terminal-moonblast-turn14.json').read_text())
    req.update(decisionMs=3500)
    engine = Lookahead(None, value_model=FlatValue())
    start = time.monotonic()
    try:
        result = search(engine, req)
        advice = plan(result, rho=.3)
    finally:
        engine.close()
    assert advice.ranked()[0][0] == 'move:moonblast'
    summary = advice.status['terminal_search']
    assert summary['ready'] and summary['simulations'] > 0
    assert summary['terminal'] > 0 and summary['failed'] == 0
    assert summary['max_tree_decisions'] >= 2  # not just a longer scripted rollout
    assert not summary['hidden_sets_exhaustive']
    assert time.monotonic() - start < 5


def test_certified_immediate_win_cannot_be_outvoted_by_incomplete_or_optimistic_leaves():
    row = {'p1': ['move:calmmind', 'move:moonblast'], 'p2': ['move:splash'],
           'payoff': [[None], [1]], 'terminal_search': {'immediate_wins': ['move:moonblast']}}
    advice = plan([row, copy.deepcopy(row)])
    assert advice.ranked()[0][0] == 'move:moonblast'
    assert advice.status['solver'] == 'sampled-world-immediate-win'


@pytest.mark.parametrize('move', ['shadowball', 'moonblast'])
def test_faster_lethal_attack_is_taken_instead_of_a_fatal_heal_or_setup(move):
    from test_baton_decisions import CHANDELURE
    from test_expert_continuation import FLOETTE, GHOLDENGO

    ours, foe = (GHOLDENGO, CHANDELURE) if move == 'shadowball' else (FLOETTE, CHOMP)
    # No future battle events are supplied. A misleading bootstrap value cannot replace the actual winning hit.
    log = [f'|switch|p1a: Me|{ours["species"]}, L50|60/175',
           f'|switch|p2a: Foe|{foe["species"]}, L50|20/100', '|turn|3',
           '|-boost|p1a: Me|spa|2', '|-boost|p1a: Me|spe|2', '|turn|4']
    req = dict(format='gen9championsbssregmc', decisionMs=1800, observationLog=log,
               samples=[{'p1': side([ours], [{'hp': .35, 'boosts': {'spa': 2, 'spe': 2}}]),
                         'p2': side([foe], [{'hp': .2}]), 'field': {}}])
    engine = Lookahead(None, value_model=FlatValue())
    try:
        advice = plan(search(engine, req), pure=True)
    finally:
        engine.close()
    chosen = advice.ranked()[0][0]
    summary = advice.status['terminal_search']
    # Flash Cannon also KOs this 20%-HP Chandelure. If the anytime certificate finds it first,
    # accepting that proven win is the contract; the test must not demand a particular winning label.
    alternative_win = move == 'shadowball' and chosen == 'move:flashcannon' and chosen in summary['immediate_wins']
    assert chosen == 'move:' + move or alternative_win
    assert summary['failed'] == 0
    assert summary['continuation_policy'] == 'public-plans-v4'


def test_playing_one_move_uses_its_own_risk_instead_of_the_largest_mixed_probability():
    row = {'p1': ['a', 'b'], 'p2': ['x', 'y'], 'payoff': [[1, -.9], [-1, 0]]}
    mixed = plan([row], rho=1)
    assert mixed.ranked()[0][0] == 'b'
    pure = plan([row], rho=1, pure=True)
    assert pure.ranked()[0][0] == 'a'
    assert pure.status['objective'] == pytest.approx(-.9)


def test_a_forced_switch_compares_all_targets_with_one_shared_budget():
    from test_expert_continuation import FLOETTE, GHOLDENGO
    from test_public_rollouts import HISTORY

    req = {'format': 'gen9championsbssregmc', 'decisionMs': 1400,
           'observationLog': HISTORY + ['|faint|p1a: Shark'],
           'samples': [{'p1': side([CHOMP, FLOETTE, GHOLDENGO], [{'hp': 0}, {'hp': 1}, {'hp': 1}]),
                        'p2': side([HIPPO]), 'field': {}}]}
    engine = Lookahead(None, value_model=FlatValue())
    start = time.monotonic()
    try:
        results = search(engine, req)
    finally:
        engine.close()
    assert set(results[0]['p1']) == {'switch:floetteeternal', 'switch:gholdengo'}
    assert results[0]['terminal_search']['ready']
    assert results[0]['terminal_search']['failed'] == 0
    assert np.isfinite(np.asarray(results[0]['payoff'], float)).all()
    assert time.monotonic() - start < 2.5


def test_our_tree_is_shared_when_hidden_opponent_items_change():
    js = r"""
const fs=require('fs'),vm=require('vm');
const sandbox={__dirname:process.cwd()+'/services/sim-tools',
 require:n=>n==='readline'?{createInterface:()=>({on:()=>{}})}:require(n),
 process:{argv:['node','worker',process.argv[1]]},module:{exports:{}}};
vm.runInNewContext(fs.readFileSync('./services/sim-tools/lookahead.js','utf8')+
 '\nmodule.exports={setUp,step,choices,State,extractChannelMessages};',sandbox);
const d=sandbox.module.exports;
const req=JSON.parse(fs.readFileSync('./tests/fixtures/terminal-moonblast-turn14.json'));
req.samples=[req.samples[0],JSON.parse(JSON.stringify(req.samples[0]))];
req.samples[1].p2.team[0].item='Choice Scarf';
const {Search}=require('./services/sim-tools/terminal_search');
const s=new Search(req,d),choices=[];
for(const st of s.states){const b=d.State.deserializeBattle(st.snapshot),o=s.observer(b,st);
 choices.push(s.treeChoice(b,o,b.p1,[]).choice);b.destroy();}
console.log(JSON.stringify({nodes:s.nodes.size,choices}));
"""
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR)], cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=10)
    result = json.loads(out.stdout)
    assert result['nodes'] == 1
    assert len(set(result['choices'])) == 1


def test_pausing_a_live_trial_does_not_turn_it_into_a_value_sample():
    js = r"""
const fs=require('fs'),vm=require('vm');
const sandbox={__dirname:process.cwd()+'/services/sim-tools',
 require:n=>n==='readline'?{createInterface:()=>({on:()=>{}})}:require(n),
 process:{argv:['node','worker',process.argv[1]]},module:{exports:{}}};
vm.runInNewContext(fs.readFileSync('./services/sim-tools/lookahead.js','utf8')+
 '\nmodule.exports={setUp,step,choices,State,extractChannelMessages};',sandbox);
const d=sandbox.module.exports;
const req=JSON.parse(fs.readFileSync('./tests/fixtures/terminal-moonblast-turn14.json'));
const {Search}=require('./services/sim-tools/terminal_search');
const s=new Search(req,d),st=s.states[0],pending=[];
const i=st.p1.findIndex(x=>x[1]==='move:calmmind');
const j=st.p2.findIndex(x=>x[1]==='move:dragonclaw');
d.observedStep=require('./services/sim-tools/public_policy').observedStep;
s.active=s.trial([0,i,j],false,pending);
const paused=!s.active.next().done;
const final=s.finish();
console.log(JSON.stringify({paused,pending:pending.length,censored:final.summary.censored,
 terminal:final.summary.terminal,unfinished:final.summary.unfinished_at_deadline,
 scored:st.cells[i][j].n}));
"""
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR)], cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=10)
    assert json.loads(out.stdout) == {'paused': True, 'pending': 0, 'censored': 0,
                                     'terminal': 0, 'unfinished': 1, 'scored': 0}


def cell_values(prior_visits, rollout_policies=None, *, stress_check=False):
    js = r"""
const fs=require('fs'),vm=require('vm');
const sandbox={__dirname:process.cwd()+'/services/sim-tools',
 require:n=>n==='readline'?{createInterface:()=>({on:()=>{}})}:require(n),
 process:{argv:['node','worker',process.argv[1]]},module:{exports:{}}};
vm.runInNewContext(fs.readFileSync('./services/sim-tools/lookahead.js','utf8')+
 '\nmodule.exports={setUp,step,choices,State,extractChannelMessages};',sandbox);
const d=sandbox.module.exports, opts=JSON.parse(process.argv[2]);
const req={...JSON.parse(fs.readFileSync('./tests/fixtures/terminal-moonblast-turn14.json')),...opts};
const {Search}=require('./services/sim-tools/terminal_search');
const s=new Search(req,d), cell=s.states[0].cells[0][0];
cell.prior=.6; cell.priorSpeed=.2;
const before=s.reply().results[0];
cell.n=1; cell.sum=-1; cell.speed=0; cell.ends=1;
const after=s.reply().results[0];
let stress=null;
if(opts.stressCheck) {
 d.observedStep=require('./services/sim-tools/public_policy').observedStep;
 let calls=0,netCalls=0;const policy=s.policy;
 s.policy=(view,name,dex)=>{if(name==='counterplay')calls++;return policy(view,name,dex);};
 s.nets.p2={prior:(v,d,c)=>c.map(()=>1/c.length),sample:()=>{netCalls++;throw Error('net erased stress policy');}};
 s.rolloutMix.p2=1;s.rolloutTemperature.p2=1;
 // Expand once and then roll out, with a real battle and the production dispatch.
 s.treeChoice=(b,obs,side)=>({choice:d.choices(b,side)[0][0],fresh:true});
 const trial=s.trial([0,0,0],false,[]);while(!trial.next().done){}
 stress={calls,netCalls,failed:s.failures};
}
console.log(JSON.stringify({before:before.payoff[0][0],after:after.payoff[0][0],speed:after.winSpeed[0][0],
 stats:after.stats[0][0],version:s.policyVersion,policies:s.rolloutPolicies,stress}));
"""
    opts = {'priorVisits': prior_visits, 'stressCheck': stress_check}
    if stress_check:
        opts.update(samples=[{'p1': side([PASSER]),
                              'p2': side([dict(HIPPO, moves=['Splash'])]), 'field': {}}],
                    observationLog=['|poke|p1|Scolipede, L50|', '|poke|p2|Hippowdon, L50|',
                                    '|switch|p1a: Scolipede|Scolipede, L50|100/100',
                                    '|switch|p2a: Hippowdon|Hippowdon, L50|100/100', '|turn|1'])
    if rollout_policies is not None:
        opts['rolloutPolicies'] = rollout_policies
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR), json.dumps(opts)], cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=20)
    return json.loads(out.stdout)


def test_named_counterplay_rollout_is_executed_even_with_a_fully_weighted_opponent_net():
    result = cell_values(4, {'p2': ['counterplay']}, stress_check=True)
    assert result['stress']['calls'] > 0
    assert result['stress']['netCalls'] == result['stress']['failed'] == 0
    assert result['version'].endswith('+public-counterplay-v3')


def test_the_value_bootstrap_keeps_its_weight_against_a_single_rollout_only_when_asked():
    plain = cell_values(0)
    assert plain['before'] == pytest.approx(.6) and plain['after'] == pytest.approx(-1)
    assert plain['stats'] == {'n': 1, 'sum': -1, 'sumSq': 0, 'prior': .6, 'speed': 0, 'priorSpeed': .2, 'ends': 1,
                              'failed': 0}
    assert plain['policies'] == {'p1': ['expert'], 'p2': ['disrupt', 'tactical']}
    assert plain['version'] == 'public-plans-v4'
    blended = cell_values(4, {'p2': ['disrupt', 'tactical', 'expert']})
    assert blended['before'] == pytest.approx(.6)
    assert blended['after'] == pytest.approx((4 * .6 - 1) / 5)
    assert blended['speed'] == pytest.approx(4 * .2 / 5)
    assert blended['policies'] == {'p1': ['expert'], 'p2': ['disrupt', 'tactical', 'expert']}
    assert blended['version'] == 'public-plans-v4+p1:expert/p2:disrupt,tactical,expert'
    with pytest.raises(subprocess.CalledProcessError):
        cell_values(0, {'p2': ['random']})


def test_predicted_replies_get_more_trials_while_a_floor_keeps_the_others_examined():
    js = r"""
const fs=require('fs'),vm=require('vm');
const sandbox={__dirname:process.cwd()+'/services/sim-tools',
 require:n=>n==='readline'?{createInterface:()=>({on:()=>{}})}:require(n),
 process:{argv:['node','worker',process.argv[1]]},module:{exports:{}}};
vm.runInNewContext(fs.readFileSync('./services/sim-tools/lookahead.js','utf8')+
 '\nmodule.exports={setUp,step,choices,State,extractChannelMessages};',sandbox);
const d=sandbox.module.exports;
const req={...JSON.parse(fs.readFileSync('./tests/fixtures/terminal-moonblast-turn14.json')),columnFloor:.2};
const {Search}=require('./services/sim-tools/terminal_search');
const s=new Search(req,d); s.rootMix=s.states[0].p1.map((_,i)=>i===0?1e6:0); s.minimumTrials=0;
const replies=s.states[0].p2.length, weights=s.states.map(st=>st.p2.map((_,j)=>j===0?1:0));
const count=(n)=>{const c={};for(let k=0;k<n;k++){const [w,i,j]=s.pickJob();s.states[w].cells[i][j].n++;
 c[j]=(c[j]||0)+1;}return c;};
const uniform=count(replies*10);
for(const st of s.states) for(const row of st.cells) for(const c of row) c.n=0;
s.setColumns(weights);
const weighted=count(replies*25);
let bad=false; try{s.setColumns([[1]]);}catch(e){bad=true;}
console.log(JSON.stringify({replies,uniform,weighted,bad}));
"""
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR)], cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=20)
    r = json.loads(out.stdout)
    replies = r['replies']
    assert replies >= 2
    assert max(r['uniform'].values()) - min(r['uniform'].values()) <= 1  # evenly covered without a prediction
    predicted, others = r['weighted']['0'], [v for k, v in r['weighted'].items() if k != '0']
    assert predicted > 2 * max(others)  # the predicted reply gets most of the budget
    assert min(others) >= 1  # the floor keeps every reply examined
    assert r['bad']


def test_column_weights_follow_the_worker_order_and_zero_dropped_replies():
    from pokesim.search.terminal import column_weights

    results = [{'p1': ['a'], 'p2': ['x', 'y', 'z'], 'payoff': [[.5, None, -.5]]},
               {'p1': ['a'], 'p2': ['x', 'y'], 'payoff': [[None, None]]}]
    seen = []

    def predictor(payoff, labels):
        seen.append(labels)
        return np.array([.25, .75])

    weights = column_weights(results, predictor)
    assert seen == [['x', 'z']]
    assert weights == [[.25, 0., .75], [1., 1.]]


def reply_with(stats, ready=True, sims=10, wins=()):
    cells = [[dict(n=n, sum=s, prior=p, speed=sp, priorSpeed=.1, ends=n, failed=f) for n, s, p, sp, f in row]
             for row in stats]
    return {'session': 'x', 'pending': [],
            'summary': {'ready': ready, 'simulations': sims, 'terminal': sims, 'learned': 1, 'censored': 0,
                        'failed': 0, 'unfinished_at_deadline': 1, 'chance_paths': 0, 'tree_nodes': 5,
                        'max_tree_decisions': 2, 'max_rollout_turns': 7, 'errors': [], 'immediate_wins': list(wins)},
            'results': [{'p1': ['a', 'b'], 'p2': ['x'], 'stats': cells, 'payoff': [[0], [0]], 'winSpeed': [[0], [0]],
                         'visits': [[0], [0]], 'invalid': [[0], [0]]}]}


def test_worker_replies_pool_rollouts_average_bootstraps_and_keep_certificates():
    from pokesim.search.terminal import merge_replies

    first = reply_with([[(3, 1., .6, .3, 0)], [(0, 0., .2, 0., 1)]], sims=10, wins=['b'])
    second = reply_with([[(1, -1., .4, 0., 0)], [(2, 2., None, .5, 0)]], sims=20)
    merged = merge_replies([first, second], prior_visits=4)
    world = merged['results'][0]
    # cell a/x: prior mean .5 with 4 pseudo visits, rollouts (3 + 1) summing to 0
    assert world['payoff'][0][0] == pytest.approx((4 * .5 + 0) / (4 + 4))
    assert world['winSpeed'][0][0] == pytest.approx((4 * .1 + .3) / 8)
    assert world['visits'][0][0] == 4 and world['invalid'][1][0] == 1
    # cell b/x: certified by the first worker outranks its rollouts
    assert world['payoff'][1][0] == 1
    assert world['stats'][1][0] == {'n': 2, 'sum': 2., 'sumSq': 0., 'prior': .2, 'speed': .5, 'priorSpeed': .1,
                                     'ends': 2, 'failed': 1}
    assert world['uncertainty'][0][0] is not None and world['uncertainty'][0][0] >= 0
    s = merged['summary']
    assert s['simulations'] == 30 and s['unfinished_at_deadline'] == 2 and s['workers'] == 2
    assert s['worker_simulations'] == [10, 20] and s['immediate_wins'] == ['b'] and s['ready']
    unready = reply_with([[(0, 0, .5, 0, 0)], [(0, 0, None, 0, 0)]], ready=False)
    assert not merge_replies([unready, unready], 0)['summary']['ready']


def test_root_progress_pools_trial_counts_and_counts_the_bootstrap_once():
    from pokesim.search.terminal import merge_replies

    first = reply_with([[(2, 0., .5, 0., 0)], [(0, 0., .5, 0., 0)]], sims=2)
    second = reply_with([[(6, 0., .5, 0., 0)], [(0, 0., .5, 0., 0)]], sims=6)
    for reply, n, total, prior in ((first, 2, .4, .3), (second, 6, 3.6, .5)):
        res = reply['results'][0]
        res['rootProgress'] = [[0.]]
        res['stats'][0][0].update(progressN=n, progressSum=total, priorProgress=prior)
    merged = merge_replies([first, second], prior_visits=4)['results'][0]
    assert merged['rootProgress'][0][0] == pytest.approx((4 * .4 + .4 + 3.6) / 12)


def test_two_workers_search_the_same_position_on_different_streams():
    req = json.loads((Path(__file__).parent / 'fixtures/terminal-moonblast-turn14.json').read_text())
    req.update(decisionMs=3000, workers=2, priorVisits=4)
    engine = Lookahead(None, value_model=FlatValue())
    try:
        result = search(engine, req)
        assert len(engine.extra_workers) == 1
        advice = plan(result, rho=.3)
    finally:
        engine.close()
    summary = advice.status['terminal_search']
    assert summary['workers'] == 2 and summary['workers_failed'] == []
    assert len(summary['worker_simulations']) == 2 and min(summary['worker_simulations']) > 0
    assert summary['simulations'] == sum(summary['worker_simulations'])
    assert advice.ranked()[0][0] == 'move:moonblast'
    assert all(np.isfinite(np.asarray(r['payoff'], float)).all() for r in result)


def test_workers_cover_the_root_cells_together_when_no_single_worker_finishes_its_pass():
    from pokesim.search.terminal import merge_replies

    def partial(start):  # a worker whose first pass reached only half the cells
        r = reply_with([[(0, 0., .4, 0., 0)], [(0, 0., None, 0., 0)]], ready=False)
        if start:
            r = reply_with([[(0, 0., None, 0., 0)], [(0, 0., .2, 0., 0)]], ready=False)
        return r
    merged = merge_replies([partial(0), partial(1)], 4)
    assert merged['summary']['ready']
    assert merged['results'][0]['payoff'] == [[.4], [.2]]
    assert not merge_replies([partial(0), partial(0)], 4)['summary']['ready']
