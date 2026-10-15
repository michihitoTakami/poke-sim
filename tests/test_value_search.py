import copy
import json
import subprocess

import numpy as np
import pytest
from test_public_rollouts import request

from pokesim.config import ROOT, SHOWDOWN_DIR
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
    assert reply['policyVersion'] == 'public-reference-v1'
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
