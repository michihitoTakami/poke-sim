import copy
import json

import pytest
from test_expert_continuation import GHOLDENGO

from pokesim.learn.compare_value import clustered_difference, observed_rows
from pokesim.live.session import BattleSession
from pokesim.value.runtime import ValueEnsemble


def test_exact_saved_p2_request_is_aligned_with_normalised_p1_prefix_and_excludes_future(tmp_path):
    session = BattleSession([GHOLDENGO])
    session.preview(['Primarina'])
    session.select([0])
    session.apply(['|switch|p1a: Gholdengo|Gholdengo, L50|175/175',
                   '|switch|p2a: Primarina|Primarina, L50|100/100', '|turn|1'])
    request = session.request()
    request['side']['id'] = 'p2'  # Original wire label; saved public log has already been normalised.
    record = dict(started='20261019-200000', source='showdown battle-1', result='win', opponent='Foo',
                  log=[*session.log, '|move|p2a: Primarina|Haze|p2a: Primarina', '|win|me'],
                  advice=[dict(turn=1, request=request)])
    path = tmp_path / 'battle.json'
    path.write_text(json.dumps(record))
    rows, info = observed_rows(tmp_path, after='20261019-190000')
    assert len(rows) == 1 and not info['skipped']
    view = rows[0]['view']
    assert view['side'] == view['request']['side']['id'] == 'p1'
    assert not view['public']['mons']['p2']['primarina']['moves']
    assert not any('Haze' in line or '|win|' in line for line in view['history'])
    assert view['request']['side']['pokemon'][0]['stats']['spa'] > 0
    changed = copy.deepcopy(record)
    changed['result'] = 'loss'
    changed['log'][-1] = '|win|foe'
    path.write_text(json.dumps(changed))
    other, _ = observed_rows(tmp_path)
    assert other[0]['view'] == view and other[0]['target'] == 0
    assert observed_rows(tmp_path, after='20261019-200000')[0] == []


def test_uncertainty_clusters_repeated_opponents_and_weights_battles_equally():
    result = clustered_difference([1., 1., 0., 0.], ['a', 'a', 'b', 'c'],
                                   {'a': 'Foo 1', 'b': 'foo1', 'c': 'Bar'})
    assert result['mean'] == pytest.approx(1 / 3)
    assert result['battles'] == 3 and result['opponent_groups'] == 2
    with pytest.raises(ValueError, match='identity missing'):
        clustered_difference([0.], ['a'], {'a': ''})


def test_unadopted_structured_model_cannot_be_loaded_as_the_live_default(tmp_path):
    (tmp_path / 'report.json').write_text(json.dumps(dict(provenance=dict(
        target='real_outcome', feature_version='observed-value-sets-v2'))))
    with pytest.raises(ValueError, match='prospective adoption'):
        ValueEnsemble(tmp_path)
