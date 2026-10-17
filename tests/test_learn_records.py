"""Actor views from our own records: what each side saw and did, with nothing from later in the game."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.learn.records import ratings, rows_from_session, rows_of, turn_actions

LOG = ['|player|p1|us|169|1118', '|player|p2|them|265|1226', '|turn|3',
       '|-mega|p2a: Golisopod|Golisopod|Golisopite', '|move|p2a: Golisopod|Sucker Punch|p1a: Skeledirge',
       '|move|p1a: Skeledirge|Torch Song|p2a: Golisopod', '|turn|4', '|cant|p1a: Skeledirge|flinch',
       '|switch|p2a: Mimikyu|Mimikyu, L50, M|100/100', '|move|p1a: Skeledirge|Torch Song|p2a: Mimikyu|[from]Dancer',
       '|turn|5', '|move|p1a: Skeledirge|Slack Off|p1a: Skeledirge|[still]', '|-fail|p1a: Skeledirge',
       '|win|them']


def test_turn_actions_mark_megas_switches_and_no_choice():
    stones = {}
    assert turn_actions(LOG, 2, stones) == {'p1': 'move:torchsong', 'p2': 'move:suckerpunch:mega'}
    assert stones == {'p2': 'golisopite'}
    assert turn_actions(LOG, 6) == {'p1': None, 'p2': 'switch:mimikyu'}  # flinched: no choice; [from] ignored
    assert turn_actions(LOG, 10) == {'p1': None, 'p2': None}  # a failed move is not a visible choice
    assert ratings(LOG) == {'p1': 1118, 'p2': 1226}


def test_rain_electro_shot_is_an_action_despite_suppressed_animation():
    log = ['|turn|4', '|move|p2a: Archaludon|Electro Shot||[still]',
           '|-boost|p2a: Archaludon|spa|1', '|-damage|p1a: Umbreon|109/201',
           '|move|p1a: Umbreon|Calm Mind|p1a: Umbreon', '|turn|5']
    assert turn_actions(log, 0) == {'p1': 'move:calmmind', 'p2': 'move:electroshot'}


def test_incomplete_record_without_a_team_or_preview_is_not_reconstructed():
    rec = {'result': 'loss', 'lineup': ['Gholdengo', 'Floette-Eternal', 'Skeledirge'],
           'log': ['|switch|p1a: Scolipede|Scolipede, L50|165/165']}
    engine = SimpleNamespace(sampler=SimpleNamespace(base_id=lambda sid: sid))
    assert rows_of('bad-roster', rec, [], engine, {}, 'train') == ([], {'missing_team_or_preview': 1})


def test_fainted_before_acting_remains_a_value_example_without_an_imitation_target(monkeypatch):
    from pokesim.learn import records
    from pokesim.learn.dataset import encode_row

    history = []
    battle = SimpleNamespace(active_pokemon=object(), opponent_active_pokemon=object())
    session = SimpleNamespace(log=history, apply=lambda lines: history.extend(lines), decision=lambda: battle)
    view = {'phase': 'move', 'history': ['|turn|1']}
    engine = SimpleNamespace(sampler=SimpleNamespace(rng=None), own_side=lambda *_: {},
                             _query=lambda _: {'observations': [{'p1': {
                                 'legal': [('move 1', 'move:shadowball')], 'view': view}}]})
    monkeypatch.setattr(records, 'opponent_world', lambda *_: {})
    monkeypatch.setattr(records, 'field_state', lambda _: {})
    lines = ['|turn|1', '|move|p2a: Charizard|Weather Ball|p1a: Gholdengo',
             '|-damage|p1a: Gholdengo|0 fnt', '|faint|p1a: Gholdengo', '|win|them']
    rows, skipped = rows_from_session('loss', session, lines, 'p1', engine, {}, {'outcome': 0.}, sides=('p1',))
    assert len(rows) == 1 and rows[0]['action'] is None and rows[0]['outcome'] == 0.
    assert rows[0]['view']['history'] == ['|turn|1']
    assert skipped['value_only'] == 1
    # No action target is manufactured from the search's recommendation or the final result.
    monkeypatch.setattr('pokesim.learn.dataset._FEATURES', object())
    monkeypatch.setattr('pokesim.learn.dataset.choices_of', lambda *_: [('move 1', 'move:shadowball')])
    assert encode_row(rows[0]) is None


def test_observe_gives_each_side_its_own_request_and_only_public_information_about_the_other():
    req = json.loads((Path(__file__).parent / 'fixtures/terminal-moonblast-turn14.json').read_text())
    js = r"""
const fs=require('fs');const req=JSON.parse(fs.readFileSync(process.argv[2]));
const child=require('child_process').spawn('node',['services/sim-tools/lookahead.js',process.argv[1]]);
let buf='';child.stdout.on('data',d=>{buf+=d;if(buf.includes('\n')){process.stdout.write(buf);child.kill();}});
child.stdin.write(JSON.stringify({observe:true,format:req.format,samples:req.samples.slice(0,1),
 observationLog:req.observationLog,sides:['p1','p2']})+'\n');
"""
    path = Path('/tmp') / 'observe-req.json'
    path.write_text(json.dumps(req))
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR), str(path)], cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=30)
    views = json.loads(out.stdout)['observations'][0]
    p1, p2 = views['p1']['view'], views['p2']['view']
    assert p1['side'] == 'p1' and p2['side'] == 'p2' and p1['phase'] == p2['phase'] == 'move'
    assert views['p1']['legal'] and views['p2']['legal']
    # Each request is the side's own; the sampled opponent set appears only in the opponent's request.
    sampled = {m['species'].lower().replace('-', '') for m in req['samples'][0]['p2']['team']}
    p2_species = {m['details'].split(',')[0].lower().replace('-', '').replace(' ', '')
                  for m in p2['request']['side']['pokemon']}
    assert all(any(s.startswith(base) for base in sampled) for s in p2_species)  # formes keep their base
    p1_species = {m['details'].split(',')[0].lower().replace('-', '').replace(' ', '')
                  for m in p1['request']['side']['pokemon']}
    assert not p1_species & p2_species
    # p1's view of p2 is public only: no moves beyond the ones the history showed, no stats.
    for mon in p1['public']['mons']['p2'].values():
        assert 'stats' not in mon and 'evs' not in mon
        assert set(mon['moves']) <= {ln.split('|')[3].lower().replace(' ', '').replace('-', '')
                                     for ln in req['observationLog'] if ln.startswith('|move|p2a')}
    assert p1['history'] == p2['history']
    assert not any(ln.startswith('|win|') for ln in p1['history'])
