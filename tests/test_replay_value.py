import gzip
import json

import pytest

from pokesim.evaluation.branch_matches import file_sha
from pokesim.evaluation.replay_value import battle_key, build, outcome, replay_split


def log(winner='Alice', future_move='Swords Dance'):
    preview = '\n'.join(f'|poke|{s}|{sp}, L50|' for s in ('p1', 'p2')
                        for sp in ('Garchomp', 'Scolipede', 'Gholdengo', 'Umbreon', 'Skeledirge', 'Gengar'))
    return ('|gametype|singles\n|player|p1|Alice|\n|player|p2|Bob|\n' + preview +
            '\n|switch|p1a: Shark|Garchomp, L50|103/202\n|switch|p2a: Ghost|Gengar, L50|100/100\n'
            '|turn|1\n|move|p1a: Shark|Dragon Claw|p2a: Ghost\n|-damage|p2a: Ghost|40/100\n'
            f'|turn|2\n|move|p1a: Shark|{future_move}|p1a: Shark\n|win|{winner}')


def setup(root):
    curated = root / 'data/curated'
    curated.mkdir(parents=True)
    (curated / 'battles.jsonl').write_text('')
    (curated / 'split_manifest.json').write_text('{"battles": {}}')
    raw = root / 'data/raw/replays'
    raw.mkdir(parents=True)
    keys = {s: next(f'gen9championsbssregmc-{i}' for i in range(1, 100) if
                    replay_split(f'gen9championsbssregmc-{i}') == s) for s in ('train', 'development')}
    for key in keys.values():
        (raw / f'{key}.json').write_text(json.dumps({'id': key, 'formatid': 'gen9championsbssregmc',
                                                   'log': f'|j|{key}\n' + log()}))
    return keys


def rows(out):
    return {s: [json.loads(line) for line in gzip.open(out / f'{s}.jsonl.gz', 'rt')]
            for s in ('train', 'development')}


def test_public_replays_export_both_perspectives_at_pre_action_turn_boundaries(tmp_path):
    setup(tmp_path)
    out = tmp_path / 'value'
    meta = build(tmp_path, out)
    data = rows(out)
    assert not set(meta['groups']['train']) & set(meta['groups']['development'])
    assert meta['rows'] == {'train': 4, 'development': 4}
    for split, samples in data.items():
        assert [r['target'] for r in samples] == [1, 0, 1, 0]
        assert sum(r['sample_weight'] for r in samples) == pytest.approx(1)
        assert all(r['observation']['request'] is None for r in samples)
        first = samples[0]['observation']['public']
        assert first['mons']['p1']['garchomp']['moves'] == []
        assert first['mons']['p1']['garchomp']['condition'] == '50/100g'
        assert samples[2]['observation']['public']['mons']['p1']['garchomp']['moves'] == ['dragonclaw']
        assert 'Alice' not in json.dumps(samples) and 'swordsdance' not in json.dumps(samples)
        assert meta['datasets'][split]['sha'] == file_sha(out / f'{split}.jsonl.gz')
    with pytest.raises(ValueError, match='overwrite'):
        build(tmp_path, out)


def test_future_moves_and_winner_do_not_change_observations(tmp_path):
    keys = setup(tmp_path)
    build(tmp_path, tmp_path / 'before')
    raw = tmp_path / 'data/raw/replays'
    for key in keys.values():
        (raw / f'{key}.json').write_text(json.dumps({'id': key, 'formatid': 'gen9championsbssregmc',
                                                   'log': f'|j|{key}\n' + log('Bob', 'Protect')}))
    build(tmp_path, tmp_path / 'after')
    a, b = rows(tmp_path / 'before'), rows(tmp_path / 'after')
    for split in a:
        assert [r['observation'] for r in a[split]] == [r['observation'] for r in b[split]]
        assert all(x['target'] == 1 - y['target'] for x, y in zip(a[split], b[split], strict=True))


def test_deduplication_and_all_own_ladder_games_are_excluded(tmp_path):
    keys = setup(tmp_path)
    raw = tmp_path / 'data/raw/replays'
    extra = ['gen9championsbssregmc-1001', 'gen9championsbssregmc-1002', 'gen9championsbssregmc-1003']
    for key in extra:
        source_log = json.loads((raw / f"{keys['train']}.json").read_text())['log']
        (raw / f'{key}.json').write_text(json.dumps({'id': key, 'formatid': 'gen9championsbssregmc',
                                                   'log': source_log}))
    (tmp_path / 'data/curated/battles.jsonl').write_text(json.dumps({
        'source': 'showdown', 'battle_id': f'showdown:battle-{extra[0]}-password'}) + '\n')
    ladder = tmp_path / 'data/raw/ladder'
    ladder.mkdir()
    (ladder / 'new.json').write_text(json.dumps({'source': f'showdown battle-{extra[1]}'}))
    meta = build(tmp_path, tmp_path / 'out')
    assert meta['excluded']['own_ladder'] == 2
    assert meta['excluded']['duplicate'] == 1
    assert not set(extra) & set(sum(meta['groups'].values(), []))


@pytest.mark.parametrize('bad', [log().replace('|win|Alice', ''), log('Nobody'),
                               log() + '\n|win|Bob', log() + '\n|turn|3'])
def test_unknown_or_ambiguous_results_are_not_losses(bad):
    with pytest.raises(ValueError):
        outcome(bad)


def test_ties_and_password_battle_ids():
    assert outcome(log().replace('|win|Alice', '|tie|')) == .5
    assert battle_key('showdown:battle-gen9championsbssregmc-12345-secret') == 'gen9championsbssregmc-12345'


def test_leaving_after_a_game_does_not_erase_the_player_or_outcome():
    assert outcome(log() + '\n|l|Bob\n|player|p2|') == 1
    assert outcome(log('Bob') + '\n|l|Alice\n|player|p1|') == 0
