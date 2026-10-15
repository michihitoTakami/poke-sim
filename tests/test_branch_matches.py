"""Terminal root trials: real outcomes, paired worlds, censoring, reproducibility and safe resume."""
import json
import subprocess

import pytest
from test_expert_continuation import CHOMP, FLOETTE, HIPPO, PASSER, side

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.evaluation.branch_matches import (
    atomic_json,
    digest,
    ensure_manifest,
    hash_seed,
    load_cases,
    load_result,
    run_case,
    summarize,
    validate_trials,
)


def worker(request, injection=""):
    js = """
const fs = require('fs'), vm = require('vm'), path = require('path');
const sandbox = {__dirname: path.dirname(process.argv[1]),
  require: n => n === 'readline' ? {createInterface: () => ({on: () => {}})} : require(n),
  process: {argv: ['node', 'worker', process.argv[2]]}, module: {exports: {}}};
const source = fs.readFileSync(process.argv[1], 'utf8');
vm.runInNewContext(source, sandbox);
vm.runInContext(process.argv[4], sandbox);
vm.runInContext(`module.exports = req => require(${JSON.stringify(path.join(sandbox.__dirname,
  'branch_matches.js'))}).evaluateBranches(req, {setUp,choices,step,policy,State});`, sandbox);
try { console.log(JSON.stringify(sandbox.module.exports(JSON.parse(process.argv[3])))); }
catch(e) { console.log(JSON.stringify({error: e.message})); }
"""
    out = subprocess.run(["node", "-e", js, str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR),
                          json.dumps(request), injection], capture_output=True, text=True, timeout=90, check=True)
    return json.loads(out.stdout)


def request(sample=None, **options):
    return {"format": "gen9championsbssregmc", "samples": [sample or {
        "p1": side([dict(CHOMP, item="")], [{"hp": 0.01, "boosts": {"atk": 6}}]),
        "p2": side([HIPPO], [{"hp": 0.05}]), "field": {}}],
        "branches": {"seeds": [0, 1], "continuations": ["greedy/greedy", "expert/disrupt"],
                     "maxTurns": 20, "actions": ["move:dragonclaw", "move:swordsdance"], **options}}


def test_terminal_targets_follow_the_fixed_root_action_and_reproduce():
    req = request()
    req["samples"] *= 2
    out = worker(req)
    validate_trials(out, 2, [0, 1], req["branches"]["continuations"])
    assert len(out["trials"]) == 16
    assert {r["outcome"] for r in out["trials"] if r["action"] == "move:dragonclaw"} == {"win"}
    assert {r["target"] for r in out["trials"] if r["action"] == "move:dragonclaw"} == {1}
    assert {r["outcome"] for r in out["trials"] if r["action"] == "move:swordsdance"} == {"loss"}
    assert out == worker(req)
    flipped = worker(request(req["samples"][0], actions=list(reversed(out["actions"]))))
    def key(r):
        return r["seed"], r["continuation"], r["action"]
    assert sorted([r for r in out["trials"] if r["world"] == 0], key=key) == sorted(flipped["trials"], key=key)


def test_unfinished_and_failed_never_get_draw_labels_or_hp_scores():
    sample = {"p1": side([FLOETTE]), "p2": side([HIPPO]), "field": {}}
    out = worker(request(sample, maxTurns=1, actions=["move:substitute", "move:missing"]))
    assert {r["outcome"] for r in out["trials"]} == {"unfinished", "failed"}
    assert all(r["target"] is None for r in out["trials"])
    assert all(r["error"] for r in out["trials"] if r["outcome"] == "failed")
    summary = summarize(out)
    assert summary["move:substitute"]["terminal_mean"] is None
    assert summary["move:substitute"]["target_bounds"] == [0, 1]


def test_real_draw_is_distinct_from_censoring():
    out = worker(request(), "step=b=>b.tie();")
    assert all(r["outcome"] == "draw" and r["target"] == 0.5 for r in out["trials"])


@pytest.mark.parametrize("injection, reason", [
    ("step=()=>{throw new Error('Choice rejected');};", "Choice rejected"),
    ("step=b=>{b.requestState='switch';};", "guard exhausted"),
])
def test_failed_simulation_cannot_become_a_terminal_training_label(injection, reason):
    out = worker(request(), injection)
    assert all(r["outcome"] == "failed" and r["target"] is None and reason in r["error"]
               for r in out["trials"])


def test_baton_pass_and_voluntary_switch_execute_actual_entry_transitions():
    sample = {"p1": side([PASSER, CHOMP], [{"hp": 1, "boosts": {"def": 2, "spe": 2}}, {"hp": 1}]),
              "p2": side([HIPPO]), "field": {}}
    out = worker(request(sample, actions=["move:batonpass", "switch:garchomp"]))
    assert not any(r["outcome"] == "failed" for r in out["trials"])
    assert all(r["turns"] > 0 for r in out["trials"])


def test_rejects_unsupported_root_and_invalid_settings():
    req = request()
    req["samples"][0]["p1"] = side([CHOMP, PASSER], [{"hp": 0}, {"hp": 1}])
    assert "normal turn-start" in worker(req)["error"]
    assert "unknown continuation" in worker(request(continuations=["missing/greedy"]))["error"]
    assert "distinct" in worker(request(seeds=[0, 0]))["error"]
    assert "positive" in worker(request(maxTurns=0))["error"]


def test_tally_keeps_unknown_mass_in_bounds_and_policy_breakdown():
    outcomes = ["win", "loss", "draw", "unfinished", "failed"]
    out = {"actions": ["a"], "trials": [{"action": "a", "continuation": "expert/disrupt", "outcome": r}
                                      for r in outcomes]}
    s = summarize(out)["a"]
    assert s["terminal_mean"] == 0.5
    assert s["target_bounds"] == [0.3, 0.7]
    assert s["by_continuation"]["expert/disrupt"]["failed"] == 1


def test_missing_and_duplicate_trials_and_false_labels_are_rejected():
    out = worker(request(seeds=[0], continuations=["greedy/greedy"]))
    validate_trials(out, 1, [0], ["greedy/greedy"])
    out["trials"].append(out["trials"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_trials(out, 1, [0], ["greedy/greedy"])
    out["trials"].pop()
    out["trials"][0]["target"] = None
    with pytest.raises(ValueError, match="label"):
        validate_trials(out, 1, [0], ["greedy/greedy"])


def test_resume_requires_identical_inputs_and_untampered_results(tmp_path):
    contract = {"worlds": 1, "seeds": [0], "continuations": ["greedy/greedy"]}
    m = ensure_manifest(tmp_path, contract, {})
    assert ensure_manifest(tmp_path, contract, {"commit": "new-provenance"}) == m
    with pytest.raises(ValueError, match="different inputs"):
        ensure_manifest(tmp_path, dict(contract, worlds=2), {})
    case = {"file": "battle.json", "turn": 2, "actions": None, "id": "test"}
    row = {"experiment_id": m["experiment_id"], "case": case,
           "result": worker(request(seeds=[0], continuations=["greedy/greedy"]))}
    row["payload_sha"] = digest(row)
    path = tmp_path / "positions/test.json"
    atomic_json(path, row)
    assert load_result(path, m, case) == row
    row["result"]["trials"][0]["turns"] += 1
    atomic_json(path, row)
    with pytest.raises(ValueError, match="modified result"):
        load_result(path, m, case)


@pytest.mark.parametrize("case", [
    {"file": "../battle.json", "turn": 1}, {"file": "battle.json", "turn": 0},
    {"file": "battle.json", "turn": True}, {"file": "battle.json", "turn": 1, "actions": []},
])
def test_invalid_position_input_is_rejected(tmp_path, case):
    p = tmp_path / "positions.json"
    p.write_text(json.dumps([case]))
    with pytest.raises(ValueError):
        load_cases(p)


def test_generation_requires_a_fixed_python_hash_seed(monkeypatch):
    for invalid in ("", "random", "-1", "4294967296"):
        monkeypatch.setenv("PYTHONHASHSEED", invalid)
        with pytest.raises(ValueError, match="fixed hash seed"):
            hash_seed()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    assert hash_seed() == 0


def test_run_case_saves_only_the_observation_prefix(monkeypatch):
    from types import SimpleNamespace

    prefix = ["|turn|2"]
    monkeypatch.setattr("pokesim.evaluation.shadow.position", lambda rec, sets, turn: (SimpleNamespace(log=prefix),
        SimpleNamespace(force_switch=False, active_pokemon=True, opponent_active_pokemon=True, last_request={})))
    monkeypatch.setattr("pokesim.search.lookahead.field_state", lambda b: {})
    reply = worker(request(seeds=[0], continuations=["greedy/greedy"]))
    calls = []
    engine = SimpleNamespace(sampler=SimpleNamespace(rng=None), own_side=lambda *a: {"known": True},
                             opponent_side=lambda *a: {"sampled": True},
                             _query=lambda q: calls.append(q) or reply)
    manifest = {"experiment_id": "test", "contract": {"world_seed": 1, "worlds": 1, "seeds": [0],
                "continuations": ["greedy/greedy"], "max_turns": 20}}
    case = {"file": "battle.json", "turn": 2, "actions": None, "id": "test"}
    row = run_case(engine, [], {}, {"log": prefix + ["|win|future"], "result": "win"}, case, manifest)
    assert row["observation"]["log"] == prefix
    assert "future" not in json.dumps(row)
    assert calls[0]["samples"] == [{"p1": {"known": True}, "p2": {"sampled": True}, "field": {}}]
