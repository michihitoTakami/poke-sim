"""Offline root-action trials with terminal labels, fixed sampled worlds and fixed continuations.

These labels describe full-information heuristic continuations in sampled worlds. They are diagnostic data,
not measured ladder win probabilities or an observation-constrained V_ref training set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.runtime.bundle import describe

if TYPE_CHECKING:
    from pokesim.search.lookahead import Lookahead

SCHEMA = 1
OUTCOMES = ("win", "loss", "draw", "unfinished", "failed")
TARGETS = {"win": 1, "loss": 0, "draw": 0.5, "unfinished": None, "failed": None}
DEFAULT_CONTINUATIONS = ["expert/tactical", "expert/disrupt"]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_seed() -> int:
    # The existing set sampler traverses Python sets (learnsets and preview members). Random.Random's seed
    # alone does not fix their ordering across fresh processes. Set this at interpreter startup, not here.
    value = os.environ.get("PYTHONHASHSEED", "")
    if not value.isdecimal() or not 0 <= int(value) <= 4294967295:
        raise ValueError("run generation with a fixed hash seed: PYTHONHASHSEED=0 uv run python -m "
                         "pokesim.evaluation.branch_matches generate ...")
    return int(value)


def tree_sha(root: Path, suffixes: set[str]) -> str:
    return digest({str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob("*"))
                   if p.is_file() and p.suffix in suffixes})


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("positions must be a nonempty JSON array")
    cases, seen = [], set()
    for row in data:
        name, turn = row["file"], row["turn"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json"):
            raise ValueError("file must be a record filename, without directories")
        if type(turn) is not int or turn < 1:
            raise ValueError("turn must be a positive integer")
        actions = row.get("actions")
        if actions is not None and (not isinstance(actions, list) or not actions or
                                    any(not isinstance(a, str) for a in actions) or len(set(actions)) != len(actions)):
            raise ValueError("actions must be distinct action labels")
        case = {"file": name, "turn": turn, "actions": actions}
        key = digest(case)[:20]
        if key in seen:
            raise ValueError("duplicate position")
        seen.add(key)
        cases.append(dict(case, id=key))
    return cases


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def ensure_manifest(out: Path, contract: dict, provenance: dict) -> dict:
    expected = digest(contract)
    path = out / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("experiment_id") != expected or saved.get("contract") != contract:
            raise ValueError("output uses different inputs/settings/code; choose a new --out directory")
        return saved
    if out.exists() and any(out.iterdir()):
        raise ValueError("output has files but no manifest; choose an empty --out directory")
    manifest = {"schema_version": SCHEMA, "experiment_id": expected, "contract": contract,
                "provenance": provenance, "purpose": "diagnostic_development",
                "information_scope": "sampled_full_information_continuations",
                "training_ready": False}
    atomic_json(path, manifest)
    return manifest


def validate_trials(reply: dict, worlds: int, seeds: list[int], continuations: list[str]) -> None:
    actions = reply["actions"]
    if not actions or len(actions) != len(set(actions)):
        raise ValueError("worker returned missing or duplicate actions")
    expected = {(w, s, c, a) for w in range(worlds) for s in seeds for c in continuations for a in actions}
    actual = []
    for row in reply["trials"]:
        key = (row["world"], row["seed"], row["continuation"], row["action"])
        actual.append(key)
        if row["outcome"] not in TARGETS or row["target"] != TARGETS[row["outcome"]]:
            raise ValueError("invalid outcome/terminal label from worker")
        if row["outcome"] == "failed" and not row["error"]:
            raise ValueError("failed trial without a reason")
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("worker returned incomplete or duplicate paired trials")


def tally(trials: list[dict]) -> dict:
    counts = Counter(row["outcome"] for row in trials)
    n = len(trials)
    terminal = counts["win"] + counts["loss"] + counts["draw"]
    known = counts["win"] + 0.5 * counts["draw"]
    unknown = counts["unfinished"] + counts["failed"]
    return {**{k: counts[k] for k in OUTCOMES}, "trials": n, "terminal": terminal,
            "terminal_mean": known / terminal if terminal else None,
            # Worst/best possible completion of every unknown trial. This is NOT a confidence interval.
            "target_bounds": [known / n, (known + unknown) / n] if n else None}


def summarize(reply: dict) -> dict:
    return {a: {**tally([r for r in reply["trials"] if r["action"] == a]),
                "by_continuation": {c: tally([r for r in reply["trials"]
                                               if r["action"] == a and r["continuation"] == c])
                                    for c in sorted({r["continuation"] for r in reply["trials"]})}}
            for a in reply["actions"]}


def load_result(path: Path, manifest: dict, case: dict) -> dict:
    row = json.loads(path.read_text())
    contract = manifest["contract"]
    if row.get("experiment_id") != manifest["experiment_id"] or row.get("case") != case:
        raise ValueError(f"result does not belong to this experiment: {path}")
    validate_trials(row["result"], contract["worlds"], contract["seeds"], contract["continuations"])
    if row.get("payload_sha") != digest({k: v for k, v in row.items() if k != "payload_sha"}):
        raise ValueError(f"modified result: {path}")
    return row


def run_case(engine: Lookahead, sets: list[dict], own_sets: dict, rec: dict, case: dict,
             manifest: dict) -> dict:
    from pokesim.evaluation.shadow import position
    from pokesim.search.lookahead import field_state

    cfg = manifest["contract"]
    # position() consumes the protocol prefix ending at |turn|N, not the later moves, sets or outcome.
    session, battle = position(rec, sets, case["turn"])
    if battle.force_switch or battle.active_pokemon is None or battle.opponent_active_pokemon is None:
        raise ValueError("only normal turn-start decisions are supported")
    world_seed = int(digest([cfg["world_seed"], case["file"], case["turn"]])[:16], 16)
    engine.sampler.rng = random.Random(world_seed)
    own = engine.own_side(battle, own_sets)
    samples = [{"p1": own, "p2": engine.opponent_side(battle), "field": field_state(battle)}
               for _ in range(cfg["worlds"])]
    branches = {"seeds": cfg["seeds"], "continuations": cfg["continuations"], "maxTurns": cfg["max_turns"]}
    if case["actions"] is not None:
        branches["actions"] = case["actions"]
    request = {"format": FORMAT, "samples": samples, "branches": branches}
    reply = engine._query(request)
    validate_trials(reply, cfg["worlds"], cfg["seeds"], cfg["continuations"])
    row = {"schema_version": SCHEMA, "experiment_id": manifest["experiment_id"], "case": case,
           "observation": {"log": session.log, "request": battle.last_request},
           "simulation_request": request, "result": reply, "summary": summarize(reply)}
    return dict(row, payload_sha=digest(row))


def generate(args) -> None:
    fixed_hash_seed = hash_seed()

    from pokesim.data.dataset import sidecar
    from pokesim.data.prior_bundle import DEFAULT, load_bundle, sampler
    from pokesim.search.lookahead import Lookahead, to_id

    cases = load_cases(args.positions)
    # Validate train-only priors before creating any artifacts; never load operational/live priors.
    prior_path = args.prior_bundle or DEFAULT
    bundle = load_bundle(prior_path)
    sets = json.loads(args.team.read_text())
    contract = {"schema_version": SCHEMA, "format": FORMAT, "positions": cases,
                "records": {c["file"]: file_sha(args.records / c["file"]) for c in cases},
                "team_sha": file_sha(args.team), "prior_sha": file_sha(prior_path),
                "prior_manifest_sha": file_sha(sidecar(prior_path)),
                "worlds": args.worlds, "world_seed": args.world_seed, "seeds": list(range(args.seeds)),
                "max_turns": args.max_turns, "continuations": args.continuations,
                "python_code_sha": tree_sha(ROOT / "src/pokesim", {".py"}),
                "worker_code_sha": tree_sha(ROOT / "services/sim-tools", {".js"}),
                "engine_dist_sha": tree_sha(SHOWDOWN_DIR / "dist", {".js", ".json"}),
                "python": platform.python_version(), "python_hash_seed": fixed_hash_seed,
                "node": subprocess.check_output(["node", "--version"], text=True).strip()}
    manifest = ensure_manifest(args.out, contract, describe())
    todo = []
    for case in cases[:args.limit]:
        path = args.out / "positions" / f"{case['id']}.json"
        if path.exists():
            load_result(path, manifest, case)
            print(f"cached {case['file']} T{case['turn']}", flush=True)
        else:
            todo.append((case, path))
    if todo:
        engine = Lookahead(sampler(bundle, args.world_seed), samples=args.worlds)
        own_sets = {engine.sampler.base_id(to_id(s["species"])): s for s in sets}
        try:
            for case, path in todo:
                rec = json.loads((args.records / case["file"]).read_text())
                row = run_case(engine, sets, own_sets, rec, case, manifest)
                atomic_json(path, row)
                print(f"saved {case['file']} T{case['turn']}: {len(row['result']['trials'])} trials -> {path}",
                      flush=True)
        finally:
            engine.close()
    report(args.out)


def report(out: Path) -> dict:
    manifest = json.loads((out / "manifest.json").read_text())
    results = []
    for case in manifest["contract"]["positions"]:
        path = out / "positions" / f"{case['id']}.json"
        if not path.exists():
            continue
        row = load_result(path, manifest, case)
        results.append({"file": case["file"], "turn": case["turn"], "actions": summarize(row["result"])})
    summary = {"experiment_id": manifest["experiment_id"], "purpose": manifest["purpose"],
               "information_scope": manifest["information_scope"], "training_ready": False,
               "completed_positions": len(results), "planned_positions": len(manifest["contract"]["positions"]),
               "positions": results}
    atomic_json(out / "summary.json", summary)
    print(f"{len(results)}/{summary['planned_positions']} positions; bounds include unfinished/failed (not CI)")
    for row in results:
        print(f"{row['file']} T{row['turn']}")
        for action, counts in row["actions"].items():
            lo, hi = counts["target_bounds"]
            print(f"  {action:28s} W/L/D {counts['win']}/{counts['loss']}/{counts['draw']} "
                  f"unfinished {counts['unfinished']} failed {counts['failed']}  bounds [{lo:.3f}, {hi:.3f}]")
    return summary


def positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="generate resumable diagnostic terminal trials")
    gen.add_argument("--positions", type=Path, required=True, help="JSON array of {file, turn, actions?}")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--records", type=Path, default=ROOT / "data/raw/ladder")
    gen.add_argument("--team", type=Path, default=ROOT / "data/my_teams/baton.json")
    gen.add_argument("--prior-bundle", type=Path, help="default: data/processed/scoped/priors.json")
    gen.add_argument("--worlds", type=positive, default=6)
    gen.add_argument("--seeds", type=positive, default=4)
    gen.add_argument("--world-seed", type=int, default=20261016)
    gen.add_argument("--max-turns", type=positive, default=200, help="additional turns including the root turn")
    gen.add_argument("--continuations", nargs="+", default=DEFAULT_CONTINUATIONS, metavar="P1/P2")
    gen.add_argument("--limit", type=positive, help="process first N positions; omit later to resume the rest")
    rep = commands.add_parser("report", help="summarize completed position files without DB or simulator")
    rep.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if len(set(args.continuations)) != len(args.continuations):
            parser.error("continuations must be distinct")
        generate(args)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
