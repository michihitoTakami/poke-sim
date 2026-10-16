"""Fixed real positions, an offline oracle over independent streams, and regret of a live-budget search."""
import json
import subprocess

import pytest

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.evaluation.positions import candidates, merge_results, objective_of, regret, stratified, summarize


def entry(turn, chosen, expected, worst, force=False):
    return {'turn': turn, 'force': force, 'chosen': chosen,
            'status': {'search': {'terminal_search': {'simulations': 1},
                                  'root_evidence': [{'label': chosen, 'expected': expected, 'worst': worst}]}}}


def test_blunder_is_the_decision_before_the_largest_drop_of_the_played_objective():
    rec = {'result': 'loss', 'versions': {'planner': {'rho': .5}},
           'advice': [entry(1, 'move:a', .8, .4), entry(2, 'move:b', .6, .6), entry(2, 'switch:x', 0, 0, True),
                      entry(3, 'move:c', -.8, -.8), entry(4, 'random', 0, 0)]}
    assert objective_of(rec['advice'][0], .5) == pytest.approx(.6)
    pool = candidates([('rec', rec)])
    assert [r['turn'] for r in pool['blunder']] == [2]
    assert pool['blunder'][0]['drop'] == pytest.approx(1.4)
    assert [r['turn'] for r in pool['loss']] == [1, 3]  # the forced switch and the random move are not positions
    assert pool['win'] == []


def test_stratified_sample_spreads_over_turn_bins():
    import random

    rows = [{'turn': t} for t in range(1, 31)]
    picked = stratified(rows, 9, random.Random(0))
    assert len(picked) == 9
    assert sum(r['turn'] <= 3 for r in picked) == 3 and sum(r['turn'] > 8 for r in picked) == 3
    assert stratified(rows, 100, random.Random(0)) == rows


def world(payoff, visits, speed=None):
    rows, cols = len(payoff), len(payoff[0])
    return {'p1': [f'a{i}' for i in range(rows)], 'p2': [f'b{j}' for j in range(cols)], 'payoff': payoff,
            'visits': visits, 'winSpeed': speed or [[0.] * cols for _ in range(rows)],
            'invalid': [[0] * cols for _ in range(rows)], 'terminal_search': {'simulations': 10, 'terminal': 9}}


def test_merged_cells_are_visit_weighted_and_prior_only_cells_are_averaged():
    a = [world([[1., .2]], [[3, 0]], [[.5, .1]])]
    b = [world([[-1., .4]], [[1, 0]], [[0., .3]])]
    merged = merge_results([a, b])
    cell = merged[0]
    assert cell['payoff'][0][0] == pytest.approx(.5)  # (3 * 1 + 1 * -1) / 4
    assert cell['winSpeed'][0][0] == pytest.approx(.375)
    assert cell['visits'][0][0] == 4
    assert cell['payoff'][0][1] == pytest.approx(.3) and cell['visits'][0][1] == 0
    assert cell['terminal_search']['simulations'] == 20 and cell['terminal_search']['streams'] == 2


def test_regret_and_summary():
    oracle = {'move:a': .5, 'move:b': .1}
    assert regret(oracle, 'move:b') == pytest.approx(.4)
    assert regret(oracle, 'move:a') == 0 and regret(oracle, 'move:z') is None
    rows = [{'kind': 'blunder', 'regret': .4, 'agree': False, 'played_regret': .4, 'simulations': 10,
             'tree_decisions': 2, 'seconds': 1.},
            {'kind': 'win', 'regret': 0., 'agree': True, 'played_regret': 0., 'simulations': 30,
             'tree_decisions': 4, 'seconds': 2.},
            {'kind': 'win', 'regret': None, 'agree': False, 'played_regret': None, 'simulations': 30,
             'tree_decisions': 4, 'seconds': 2.}]
    s = summarize(rows)
    assert s['scored'] == 2 and s['regret'] == pytest.approx(.2) and s['agree'] == .5
    assert s['by_kind']['blunder']['regret'] == pytest.approx(.4) and s['by_kind']['win']['agree'] == 1


def test_a_stream_changes_the_search_session_and_its_absence_keeps_the_live_id():
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
const ids=[new Search(req,d).id,new Search({...req,stream:0},d).id,new Search({...req,stream:1},d).id,
 new Search({...req,stream:2},d).id];
console.log(JSON.stringify(ids));
"""
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR)], cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=20)
    live, zero, one, two = json.loads(out.stdout)
    assert live == zero
    assert len({live, one, two}) == 3
