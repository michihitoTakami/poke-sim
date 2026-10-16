"""An exported policy net gives the same probabilities in the JS worker as in torch."""
import json
import subprocess

import numpy as np
import torch

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.learn.features import Features
from pokesim.learn.model import PolicyNet, export_json, load_checkpoint, save_checkpoint
from test_policy_feature_parity import observe


def test_exported_net_matches_torch_on_fixture_views(tmp_path):
    features = Features.load()
    torch.manual_seed(3)
    model = PolicyNet(len(features.state_names), len(features.action_names), hidden=32, embed=16,
                      action_hidden=8, scorer_hidden=8)
    export_json(tmp_path / 'net.json', model, {'head': 'opp', 'feature_version': 'policy-features-v1'})
    save_checkpoint(tmp_path / 'net.pt', model, {'head': 'opp'})
    reloaded, _ = load_checkpoint(tmp_path / 'net.pt')
    req = json.loads((ROOT / 'tests/fixtures/terminal-moonblast-turn14.json').read_text())
    rows = [{'view': v[side]['view'], 'legal': v[side]['legal']} for v in observe(req) for side in ('p1', 'p2')]
    out = subprocess.run(['node', 'services/sim-tools/policy_net_dump.js', str(SHOWDOWN_DIR), FORMAT,
                          str(tmp_path / 'net.json')], input=''.join(json.dumps(r) + '\n' for r in rows), cwd=ROOT,
                         text=True, capture_output=True, check=True, timeout=120)
    js = [json.loads(line) for line in out.stdout.splitlines()]
    for row, probs in zip(rows, js, strict=True):
        state, actions = features.vectors(row['view'], row['legal'])
        mask = torch.ones(1, len(row['legal']), dtype=torch.bool)
        with torch.no_grad():
            logits = reloaded(torch.as_tensor(state)[None], torch.as_tensor(actions)[None], mask)
        expected = torch.softmax(logits, dim=-1)[0].numpy()
        assert np.abs(expected - np.asarray(probs)).max() < 1e-5
        assert abs(sum(probs) - 1) < 1e-6


def test_search_uses_the_net_for_priors_and_rollouts(tmp_path):
    import time

    from pokesim.search.lookahead import Lookahead
    from pokesim.search.planner import plan
    from pokesim.search.terminal import search
    from test_terminal_search import FlatValue

    features = Features.load()
    torch.manual_seed(5)
    model = PolicyNet(len(features.state_names), len(features.action_names), hidden=32, embed=16,
                      action_hidden=8, scorer_hidden=8)
    export_json(tmp_path / 'opp.json', model, {'head': 'opp', 'feature_version': 'policy-features-v1'})
    req = json.loads((ROOT / 'tests/fixtures/terminal-moonblast-turn14.json').read_text())
    req.update(decisionMs=3000, policyNets={'opp': str(tmp_path / 'opp.json')}, rolloutMix={'opp': 1.0},
               rolloutNetTurns=4, priorVisits=4)
    engine = Lookahead(None, value_model=FlatValue())
    start = time.monotonic()
    try:
        result = search(engine, req)
        advice = plan(result, rho=.3)
    finally:
        engine.close()
    summary = advice.status['terminal_search']
    assert summary['policy_nets'] == {'opp': __import__('hashlib').sha256(
        (tmp_path / 'opp.json').read_bytes()).hexdigest()}
    assert summary['puct'] == 1.5 and '+nets:opp=' in summary['continuation_policy']
    assert summary['simulations'] > 50 and summary['failed'] == 0 and summary['max_tree_decisions'] >= 2
    for world in result:
        assert world['p2Prior'] is not None and len(world['p2Prior']) == len(world['p2'])
        assert abs(sum(world['p2Prior']) - 1) < 1e-6
    assert advice.ranked()[0][0] == 'move:moonblast'
    assert time.monotonic() - start < 5
