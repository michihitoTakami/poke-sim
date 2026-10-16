"""Actor views from our own records: what each side saw and did, with nothing from later in the game."""
import json
import subprocess
from pathlib import Path

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.learn.records import ratings, turn_actions

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
