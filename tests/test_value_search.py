import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_public_rollouts import request

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.search.lookahead import Lookahead
from pokesim.value.runtime import ValueEnsemble


def query(req):
    result = subprocess.run(['node', str(ROOT / 'services/sim-tools/lookahead.js'), str(SHOWDOWN_DIR)],
                            input=json.dumps(req) + '\n', text=True, capture_output=True, check=True, timeout=60)
    reply = json.loads(result.stdout)
    assert 'error' not in reply, reply
    return reply


def value_request():
    req = request()
    branch = req.pop('branches')
    req.update(valueLeaves=True, observationLog=branch['observationLog'], seeds=1, rollout=0,
               continuations=['expert/tactical', 'expert/disrupt'])
    return req


def test_nonterminal_leaf_uses_model_and_terminal_uses_exact_outcome():
    ensemble = ValueEnsemble.__new__(ValueEnsemble)
    ensemble.version = {'reference': {'version': 'public-reference-v1'}}
    ensemble.probabilities = lambda obs: np.full(len(obs), .8)
    leaves = [{'i': 0, 'j': 0, 'continuation': 'expert/tactical', 'terminal': None,
               'observation': {}, 'hpDiagnostic': -.4},
              {'i': 1, 'j': 0, 'continuation': 'expert/tactical', 'terminal': 1,
               'observation': None, 'hpDiagnostic': .95},
              {'i': 2, 'j': 0, 'continuation': 'expert/tactical', 'terminal': 0,
               'observation': None, 'hpDiagnostic': -.95}]
    reply = {'policyVersion': 'public-reference-v1',
             'results': [{'p1': ['a', 'b', 'c'], 'p2': ['x'], 'invalid': [[0], [0], [0]], 'leaves': leaves}]}
    result = ensemble.apply(reply)[0]
    assert np.allclose(result['payoff'], [[.6], [1], [-1]])
    assert result['resolved'] == [[0], [1], [1]]
    assert result['value_status']['learned'] == 1 and result['value_status']['terminal'] == 2
    assert result['hpDiagnostic'][0][0] == -.4


def test_worker_returns_actor_observations_and_paired_complete_passes():
    reply = query(value_request())
    assert reply['policyVersion'] == 'public-reference-v2'
    for result in reply['results']:
        assert result['passes'] == [2, 2]
        assert not np.asarray(result['invalid']).any(), result['value_errors']
        for leaf in result['leaves']:
            if leaf['terminal'] is None:
                obs = leaf['observation']
                assert obs['phase'] == 'move' and obs['side'] == obs['request']['side']['id'] == 'p1'
                assert all('stats' not in m for m in obs['public']['mons']['p2'].values())
                assert not any(h.startswith('|win|') for h in obs['history'])


def test_unrevealed_opponent_changes_cannot_change_an_identical_public_leaf():
    req = value_request()
    req['samples'][0]['p2']['team'][0]['moves'] = ['Protect', 'Earthquake', 'Yawn', 'Slack Off']
    req['samples'][0]['p2']['team'][0]['item'] = 'Choice Band'
    other = copy.deepcopy(req['samples'][0])
    other['p2']['team'][0].update(nature='Timid', item='Choice Specs')
    req['samples'].append(other)
    reply = query(req)
    observations = []
    for res in reply['results']:
        i, j = res['p1'].index('move:protect'), res['p2'].index('move:protect')
        observations.append(next(r['observation'] for r in res['leaves'] if r['i'] == i and r['j'] == j))
    assert observations[0] == observations[1]


def test_incoming_transition_updates_public_identity_without_exposing_sampled_preview():
    req = value_request()
    req['samples'][0]['p1']['incoming'] = 1
    reply = query(req)
    res = reply['results'][0]
    for leaf in res['leaves']:
        if leaf['terminal'] is None and res['p1'][leaf['i']] == 'move:protect':
            obs = leaf['observation']
            assert obs['public']['active']['p1'] == 'scolipede'
            assert obs['public']['preview']['p2'] == ['Hippowdon, L50', 'Gholdengo, L50', 'Gengar, L50']
            break
    else:
        pytest.fail('no incoming continuation leaf generated')


def test_unsupported_full_information_continuations_are_rejected():
    req = value_request()
    req['continuations'] = ['expert@6/tactical']
    out = subprocess.run(['node', str(ROOT / 'services/sim-tools/lookahead.js'), str(SHOWDOWN_DIR)],
                         input=json.dumps(req) + '\n', text=True, capture_output=True, check=True, timeout=30)
    assert 'frozen public reference' in json.loads(out.stdout)['error']


def test_a_stuck_worker_is_terminated_instead_of_hanging_the_battle():
    engine = Lookahead.__new__(Lookahead)
    engine.query_timeout = .05
    engine.proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    with pytest.raises(TimeoutError, match='deadline'):
        engine._query({})
    assert engine.proc.poll() is not None


def test_reconstructed_names_update_public_hp_and_moves_for_the_same_active_slot():
    req = value_request()
    req['observationLog'] = [h.replace('p2a: Hippo', 'p2a: nicknamed foe') for h in req['observationLog']]
    res = query(req)['results'][0]
    i = res['p1'].index('move:dragonclaw')
    j = res['p2'].index('move:earthquake')
    obs = next(r['observation'] for r in res['leaves'] if r['i'] == i and r['j'] == j)
    mon = obs['public']['mons']['p2']['hippowdon']
    assert mon['condition'] != '100/100'
    assert 'earthquake' in mon['moves']


def test_trapping_rejection_retries_from_the_engines_visible_request_update():
    js = """
const {observedStep}=require('./services/sim-tools/public_policy');
const side={id:'p1',activeRequest:{active:[{maybeTrapped:true}]}};
const battle={p1:side,p2:{id:'p2'}};let attempts=0,retries=0;
const step=(b,a,c)=>{attempts++;if(a==='switch 2') {
 b.p1.activeRequest.active[0]={trapped:true};
 throw Error("Choice rejected: p1 switch 2: Can't switch: The active Pokémon is trapped");}
 if(c!=='move 3') throw Error('opponent choice was changed');};
observedStep(battle,'switch 2','move 3',step,s=>{
 if(!s.activeRequest.active[0].trapped)throw Error('missing observation');
 retries++;return 'move 1';});
let fixedRejected=false;try{observedStep(battle,'switch 2','move 3',step,()=> 'move 1',[]);}
catch(e){fixedRejected=true;}
console.log(JSON.stringify({attempts,retries,fixedRejected}));
"""
    result = subprocess.run(['node', '-e', js], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {'attempts': 3, 'retries': 1, 'fixedRejected': True}


def test_showdown_p2_history_is_normalized_to_the_models_own_p1_perspective():
    engine = Lookahead.__new__(Lookahead)
    engine.own_side = lambda *args: {'team': 'ours'}
    engine.opponent_side = lambda *args: {'team': 'theirs'}
    engine.n_samples, engine.seeds, engine.rollout = 1, 1, 0
    engine.win_decay, engine.search = None, {}
    engine._query = lambda req: req
    engine.value_model = SimpleNamespace(apply=lambda req: req)
    battle = SimpleNamespace(weather={}, fields={}, turn=1)
    history = ['|poke|p2|Gholdengo, L50|', '|switch|p2a: Gold|Gholdengo, L50|100/100', '|turn|1']
    req = engine.query(battle, {}, log=history, our_role='p2', value=True)
    assert req['samples'][0]['p1']['team'] == 'ours'
    assert req['observationLog'][0] == '|poke|p1|Gholdengo, L50|'
    assert req['observationLog'][1].startswith('|switch|p1a: Gold|')
    assert history[0].startswith('|poke|p2|')
