"""Take a real finishing move; never certify lucky, blocked, or non-terminal KOs as a match win."""
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from test_expert_continuation import CHOMP, HIPPO, side

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
