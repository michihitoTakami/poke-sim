import copy

import numpy as np
import pytest
import torch
from test_value_training import dex, observation

from pokesim.learn.value_dataset import normalise
from pokesim.value.model import create_model, load_checkpoint
from pokesim.value.structured import Encoder, weather_clock


def fixture():
    data = dex()
    data['species'].append(dict(data['species'][0], id='floetteeternal', name='Floette-Eternal',
                                baseSpecies='Floette-Eternal', types=['Fairy']))
    data['moves'].append(dict(data['moves'][0], id='moonblast', category='Special', basePower=95, type='Fairy'))
    obs = observation(True)
    for species in ['Gholdengo', 'Floette-Eternal']:
        obs['request']['side']['pokemon'].append(dict(details=species, condition='100/100', active=False,
                                                     moves=['recover', 'moonblast']))
        obs['public']['mons']['p1'][species] = dict(species=species, condition='100/100', moves=['recover'])
    obs['request']['active'][0]['moves'] = [dict(id='recover', pp=0, maxpp=8, disabled=True),
                                          dict(id='moonblast', pp=16, maxpp=16, disabled=False)]
    obs['request']['side']['pokemon'][0]['moves'] = ['recover', 'moonblast']
    return Encoder(data), obs


def test_recipient_status_and_per_move_pp_do_not_alias():
    enc, obs = fixture()
    changed = copy.deepcopy(obs)
    obs['request']['side']['pokemon'][1]['condition'] += ' par'
    changed['request']['side']['pokemon'][2]['condition'] += ' par'
    assert not np.array_equal(enc.encode(obs)[1], enc.encode(changed)[1])
    changed = copy.deepcopy(obs)
    changed['request']['active'][0]['moves'] = [dict(id='recover', pp=8, maxpp=8, disabled=False),
                                              dict(id='moonblast', pp=0, maxpp=16, disabled=True)]
    assert not np.array_equal(enc.encode(obs)[1], enc.encode(changed)[1])


def test_nested_model_ignores_roster_and_move_slot_order_and_reloads(tmp_path):
    torch.manual_seed(13)
    enc, obs = fixture()
    p, q, _, _, schema = enc.encode(obs)
    model = create_model(schema, 64)
    # Test the private branch too; it starts at zero for stable public pretraining.
    torch.nn.init.normal_(model.private_head[-1].weight)
    expected = model(torch.tensor(p[None]), torch.tensor(q[None]))
    permuted_p, permuted_q = p.copy(), q.copy()
    for b in schema['sets']['blocks']:
        data = permuted_p if b['branch'] == 'public' else permuted_q
        tokens = data[b['start']:b['start'] + b['width'] * b['count']].reshape(b['count'], b['width'])
        tokens[:] = tokens[::-1].copy()
        if 'preview' not in b['label']:
            moves = tokens[:, schema['sets']['move_start']:].reshape(b['count'], 4, schema['sets']['move_width'])
            moves[:] = moves[:, ::-1].copy()
    assert torch.allclose(model(torch.tensor(permuted_p[None]), torch.tensor(permuted_q[None])), expected, atol=1e-7)
    path = tmp_path / 'model.pt'
    torch.save(dict(schema_version=1, model_config=model.config, state_dict=model.state_dict()), path)
    loaded, _ = load_checkpoint(path)
    assert torch.equal(loaded(torch.tensor(p[None]), torch.tensor(q[None])), expected)
    # Arbitrary outcome/provenance and opponent private fields cannot alter features.
    obs.update(target=0, root_action='move:recover', source='future')
    obs['request']['opponent'] = {'moves': ['secret'], 'stats': {'spe': 999}}
    assert np.array_equal(enc.encode(obs)[0], p)
    assert np.array_equal(enc.encode(obs)[1], q)


def test_missing_private_request_is_masked_and_not_an_invented_team():
    enc, obs = fixture()
    obs['request'] = None
    p, q, _, _, schema = enc.encode(obs)
    assert not q.any()
    model = create_model(schema, 64)
    assert torch.equal(model(torch.tensor(p[None])), model(torch.tensor(p[None]), torch.tensor(q[None])))


def test_weather_clock_counts_public_upkeep_not_repeated_weather_as_a_new_start():
    _, obs = fixture()
    obs['public']['weather'] = 'SunnyDay'
    obs['history'] = ['|switch|p2a: Y|Charizard-Mega-Y, L50|100/100', '|turn|1',
                      '|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Y']
    assert weather_clock(obs)['remaining_min'] == 5 / 8
    obs['history'] += ['|-weather|SunnyDay|[upkeep]'] * 4
    assert weather_clock(obs)['remaining_min'] == weather_clock(obs)['remaining_max'] == 1 / 8
    # A later switch cannot make the source acquire another Pokemon's item/species.
    obs['history'] += ['|switch|p2a: Ninetales|Ninetales, L50|100/100']
    assert weather_clock(obs)['remaining_max'] == 1 / 8
    obs['history'] += ['|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Ninetales']
    assert weather_clock(obs)['remaining_min'] == 5 / 8 and weather_clock(obs)['remaining_max'] == 1
    obs['public']['weather'] = ''
    assert not weather_clock(obs)['known']
    obs['public']['weather'] = 'SunnyDay'
    obs.pop('history')
    assert not weather_clock(obs)['known']


def test_p2_normalisation_swaps_public_history_without_swapping_nicknames():
    _, obs = fixture()
    obs['side'], obs['request']['side']['id'] = 'p2', 'p2'
    obs['history'] = ['|switch|p2a: p1nickname|Gholdengo, L50|100/100',
                      '|-weather|SunnyDay|[of] p1a: Y', '|poke|p1|Gholdengo']
    result = normalise(obs)
    assert result['history'] == ['|switch|p1a: p1nickname|Gholdengo, L50|100/100',
                                 '|-weather|SunnyDay|[of] p2a: Y', '|poke|p2|Gholdengo']
    assert obs['history'][0].startswith('|switch|p2a:')


def test_model_rejects_an_unknown_architecture():
    from pokesim.value.model import model_from_config

    with pytest.raises(ValueError, match='architecture'):
        model_from_config(dict(architecture='unknown'))
