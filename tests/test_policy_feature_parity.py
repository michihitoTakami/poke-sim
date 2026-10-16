"""The JS and Python policy features agree exactly on the same views (the net sees in play what it saw in training)."""
import json
import subprocess
from pathlib import Path

import numpy as np

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.learn.features import Features


def observe(req, sides=('p1', 'p2')):
    js = r"""
const fs=require('fs');const req=JSON.parse(fs.readFileSync(process.argv[2]));
const child=require('child_process').spawn('node',['services/sim-tools/lookahead.js',process.argv[1]]);
let buf='';child.stdout.on('data',d=>{buf+=d;if(buf.includes('\n')){process.stdout.write(buf);child.kill();}});
child.stdin.write(JSON.stringify({observe:true,format:req.format,samples:req.samples,
 observationLog:req.observationLog,sides:JSON.parse(process.argv[3])})+'\n');
"""
    path = Path('/tmp') / 'parity-req.json'
    path.write_text(json.dumps(req))
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR), str(path), json.dumps(list(sides))], cwd=ROOT,
                         text=True, capture_output=True, check=True, timeout=60)
    return json.loads(out.stdout)['observations']


def js_vectors(rows):
    out = subprocess.run(['node', 'services/sim-tools/policy_features_dump.js', str(SHOWDOWN_DIR), FORMAT],
                         input=''.join(json.dumps(r) + '\n' for r in rows), cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=120)
    return [json.loads(line) for line in out.stdout.splitlines()]


def compare(rows, tolerance=1e-5):
    features = Features.load()
    js = js_vectors(rows)
    worst = 0.
    for row, expected in zip(rows, js, strict=True):
        state, actions = features.vectors(row['view'], row['legal'])
        assert state.shape == (len(features.state_names),)
        assert actions.shape == (len(row['legal']), len(features.action_names))
        worst = max(worst, float(np.abs(state - np.asarray(expected['state'], np.float32)).max()),
                    float(np.abs(actions - np.asarray(expected['actions'], np.float32)).max()))
        assert np.isfinite(state).all() and np.isfinite(actions).all()
    assert worst < tolerance, worst
    return worst


def test_fixture_views_match_between_js_and_python():
    req = json.loads((ROOT / 'tests/fixtures/terminal-moonblast-turn14.json').read_text())
    rows = [{'view': v[side]['view'], 'legal': v[side]['legal']} for v in observe(req) for side in ('p1', 'p2')]
    assert len(rows) == 4
    compare(rows)


def test_features_describe_the_position():
    req = json.loads((ROOT / 'tests/fixtures/terminal-moonblast-turn14.json').read_text())
    view = observe(req)[0]['p1']['view']
    features = Features.load()
    state = features.state(view)
    assert state['foe.hp'] < .05 and state['me.outgoing'] == 2 and state['me.faster'] == 0  # the finishing blow
    moonblast = features.action(view, 'move 1', 'move:moonblast')
    calm = features.action(view, 'move 4', 'move:calmmind')
    assert moonblast['move.kills'] == 1 and moonblast['move.stab'] == 1 and moonblast['move.effectiveness'] == .5
    assert calm['move.setup'] == 1 and calm['move.boost_spa'] == .5 and calm['move.damage'] == 0
