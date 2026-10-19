"""Generate public terminal branches, learn action ranks and a value for the same continuation policy.

The inspected #37 games are development data. All positions, worlds, seeds and actions of a source battle
share its frozen split. Executed manual moves are candidate interventions, never automatic correct labels.
No human game outcome is used as a per-action target. Unfinished/failed trials do not become losses.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.evaluation.branch_matches import (
    atomic_json,
    digest,
    ensure_manifest,
    file_sha,
    hash_seed,
    load_result,
    normalize_cases,
    run_case,
    tree_sha,
)

AUDIT = ROOT / "docs/audits/20261018-post-update-analysis.json"
LEGACY_CONTINUATIONS = ["planned/tactical", "planned/disrupt", "planned/expert"]
CONTINUATIONS = [*LEGACY_CONTINUATIONS, "planned/counterplay"]


def confident_pairs(trials, actions, minimum_worlds=4):
    """Paired world bootstrap, retaining policy diversity. Seeds within a world are not independent evidence.

    A single fortunate crit or one weak opponent controller cannot teach a confident action preference.
    Censored/failed trials and incomplete comparisons produce no preference, rather than a loss label.
    """
    by_action = {a: {} for a in actions}
    for t in trials:
        key = (t['world'], t['seed'], t['continuation'])
        bucket = by_action[t['action']]
        if key in bucket:
            raise ValueError('duplicate paired branch trial')
        bucket[key] = t['target']
    weights = np.zeros((len(actions), len(actions)), np.float32)
    for i, a in enumerate(actions):
        for j, b in enumerate(actions):
            if i == j or set(by_action[a]) != set(by_action[b]):
                continue
            keys = sorted(by_action[a])
            if any(by_action[a][k] is None or by_action[b][k] is None for k in keys):
                continue
            worlds = sorted({k[0] for k in keys})
            if len(worlds) < minimum_worlds:
                continue
            delta = {k: by_action[a][k] - by_action[b][k] for k in keys}
            # No teaching preference if any named opponent policy reverses its sign on average.
            if any(np.mean([v for k, v in delta.items() if k[2] == name]) < 0
                   for name in {k[2] for k in keys}):
                continue
            per_world = np.array([np.mean([v for k, v in delta.items() if k[0] == w]) for w in worlds])
            rng = np.random.default_rng(0)
            lower = np.quantile(rng.choice(per_world, (2000, len(worlds))).mean(1), .05)
            if lower > 0:
                weights[i, j] = per_world.mean()
    return weights.tolist()


def battle_identity(value: str) -> str:
    return re.sub(r"^showdown[ :](?:battle-)?", "", value)


def development_files() -> set[str]:
    audit = json.loads(AUDIT.read_text())
    return {r["file"] for r in audit["groups"]["new"]["records"]}


def development_battles() -> set[str]:
    return {r["battle_id"] for r in json.loads(AUDIT.read_text())["new_battles"]}


def select_positions(records: Path, limit: int, development: set[str]) -> list[dict]:
    """Prefer disagreements and large evaluation drops in losses; deterministic battle-level sampling."""
    from pokesim.evaluation.positions import candidates
    from pokesim.live.provenance import actual_lineup, execution_rows

    found = []
    counts = Counter()
    seen = set()
    for path in sorted(records.glob("*.json"), key=lambda p: digest(p.name)):
        rec = json.loads(path.read_text())
        bid = rec.get("source")
        if not bid or bid in seen or rec.get("result") not in {"win", "loss"} or not rec.get("preview"):
            continue
        split = "development" if path.name in development else "train"
        # The audit's previously inspected games are the fixed development cohort; later live games stay out.
        if rec.get("started", "") > "20261018-021838" or counts[split] >= limit or not actual_lineup(rec)["complete"]:
            continue
        executions = execution_rows(rec)
        correction = next(
            (e for e in executions if e["executed"] and e["suggested"] and e["executed"] != e["suggested"]), None
        )
        drops = candidates([(path.stem, rec)])["blunder"]
        turn = correction["turn"] if correction else drops[0]["turn"] if drops else 1
        if f"|turn|{turn}" not in rec["log"]:
            continue
        found.append({"file": path.name, "turn": turn, "split": split})
        counts[split] += 1
        seen.add(bid)
    return found


def generate(args):
    from pokesim.data.dataset import sidecar
    from pokesim.data.prior_bundle import load_bundle, sampler
    from pokesim.search.lookahead import Lookahead, to_id

    fixed_hash_seed = hash_seed()
    positions = (
        json.loads(args.positions.read_text())
        if args.positions
        else select_positions(args.records, args.limit, development_files())
    )
    dev = development_files()
    # Freeze identity/split across every position, including explicitly supplied candidate lists.
    engine_sha = tree_sha(SHOWDOWN_DIR / "dist", {".js", ".json"})
    assignments, records, cases = {}, {}, {"train": [], "development": [], "calibration": []}
    for row in positions:
        split = row["split"]
        if split not in cases or (row["file"] in dev and split != "development"):
            raise ValueError("inspected #37 battles must stay in development; no test tuning")
        rec = json.loads((args.records / row["file"]).read_text())
        bid = rec.get("source") or row["file"]
        if bid in development_battles() and split != "development":
            raise ValueError("a renamed inspected battle must also stay in development")
        if bid in assignments and assignments[bid] != split:
            raise ValueError("branches of the same battle cannot cross splits")
        assignments[bid] = split
        records[row["file"]] = (bid, file_sha(args.records / row["file"]))
        cases[split].append({k: v for k, v in row.items() if k != "split"})
    bundle = load_bundle(args.priors)
    prior_meta = json.loads(sidecar(args.priors).read_text())
    # The base prior may use external training replays; it must never contain a development source battle.
    if prior_meta.get("split") != "train":
        raise ValueError("branch worlds require a frozen train-only prior")
    if set(map(battle_identity, prior_meta.get("records", {}))) & {
        battle_identity(b) for b, s in assignments.items() if s in {"development", "calibration"}
    }:
        raise ValueError("development battle leaked into the set prior")
    split_contract = {
        "assignments": assignments,
        "records": records,
        "inspected_development": sorted(dev),
        "no_unseen_test": True,
    }
    split_path = args.out / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split_contract:
        raise ValueError("source inputs changed; use a new output directory")
    atomic_json(split_path, split_contract)
    sets = json.loads(args.team.read_text())
    engine = Lookahead(sampler(bundle, args.seed))
    engine.query_timeout = 300.0
    own = {engine.sampler.base_id(to_id(s["species"])): s for s in sets}
    try:
        for split, items in cases.items():
            if not items:
                continue
            normalized = normalize_cases(items)
            source_records = {records[c["file"]][0]: records[c["file"]][1] for c in normalized}
            cfg = {
                "schema_version": 1,
                "format": FORMAT,
                "engine_dist_sha": engine_sha,
                "python_hash_seed": fixed_hash_seed,
                "split": split,
                "positions": normalized,
                "information_scope": "public_observation",
                "source_scope": {"split": split, "records": source_records, "split_manifest_sha": file_sha(split_path)},
                "battle_ids": {c["file"]: records[c["file"]][0] for c in normalized},
                "records": {c["file"]: records[c["file"]][1] for c in normalized},
                "prior_sha": file_sha(args.priors),
                "prior_manifest_sha": file_sha(sidecar(args.priors)),
                "team_sha": file_sha(args.team),
                "worlds": args.worlds,
                "seeds": list(range(args.seeds)),
                "world_seed": args.seed,
                "max_turns": args.max_turns,
                "continuations": CONTINUATIONS,
                "policy_nets": {"ours": str(args.continuation_net.resolve())} if args.continuation_net else {},
                "continuation_net_sha": file_sha(args.continuation_net) if args.continuation_net else None,
                "python_code_sha": tree_sha(ROOT / "src/pokesim", {".py"}),
                "worker_code_sha": tree_sha(ROOT / "services/sim-tools", {".js"}),
            }
            directory = args.out / split
            manifest = ensure_manifest(directory, cfg, {"purpose": "terminal-branch-learning"})
            for case in normalized:
                path = directory / "positions" / f"{case['id']}.json"
                if path.exists():
                    load_result(path, manifest, case)
                    continue
                rec = json.loads((args.records / case["file"]).read_text())
                try:
                    from pokesim.live.provenance import actual_lineup

                    if not actual_lineup(rec)["complete"]:
                        raise ValueError("actual selected three are not confirmed; keep this game for imitation only")
                    result = run_case(engine, sets, own, rec, case, manifest)
                except ValueError as e:
                    print(f"skip {case['file']} T{case['turn']}: {e}", flush=True)
                    continue
                from pokesim.live.provenance import execution_rows

                execution = next((r for r in execution_rows(rec) if r["turn"] == case["turn"]), {})
                result["root_provenance"] = execution
                result["payload_sha"] = digest({k: v for k, v in result.items() if k != "payload_sha"})
                atomic_json(path, result)
                tally = Counter(t["outcome"] for t in result["result"]["trials"])
                print(f"{split} {case['file']} T{case['turn']}: {dict(tally)}", flush=True)
    finally:
        engine.close()


def examples(out: Path):
    """Position-level action means and non-intervened continuation values, with battle weights."""
    ranks, values, stats, groups = {}, {}, Counter(), {}
    for split in ("train", "development", "calibration"):
        directory = out / split
        if not (directory / "manifest.json").exists():
            ranks[split], values[split], groups[split] = [], [], set()
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        cfg = manifest["contract"]
        if manifest["experiment_id"] != digest(cfg):
            raise ValueError("modified branch manifest")
        if (cfg["continuations"] not in (LEGACY_CONTINUATIONS, CONTINUATIONS) or
                manifest["information_scope"] != "public_observation"):
            raise ValueError("unsupported continuation target")
        ranks[split], values[split], groups[split] = [], [], set()
        for case in cfg["positions"]:
            path = directory / "positions" / f"{case['id']}.json"
            if not path.exists():
                stats["missing_positions"] += 1
                continue
            result = load_result(path, manifest, case)
            bid = cfg["battle_ids"][case["file"]]
            if case["file"] in development_files() and split != "development":
                raise ValueError("inspected #37 position is outside development")
            groups[split].add(bid)
            if result["result"].get("continuationNet") != cfg.get("continuation_net_sha"):
                raise ValueError("value trajectories came from a different continuation policy")
            trials = result["result"]["trials"]
            by_action = {a: [t for t in trials if t["action"] == a] for a in result["result"]["actions"]}
            # Censoring cannot make a stalled action appear better by averaging only its completed wins.
            if all(t["target"] is not None for t in trials):
                obs = next(d["observation"] for d in trials[0]["trajectory"] if d["side"] == "p1")
                ranks[split].append(
                    {
                        "group": bid,
                        "position": case["id"],
                        "view": obs,
                        "actions": list(by_action),
                        "targets": [float(np.mean([t["target"] for t in ts])) for ts in by_action.values()],
                        "pair_weights": confident_pairs(trials, list(by_action)),
                        "policy_version": result['result']['policyVersion'],
                        "root_provenance": result.get("root_provenance", {}),
                        "baseline_action": next(
                            d.get("policyRecommendation") for d in trials[0]["trajectory"] if d["side"] == "p1"
                        ),
                    }
                )
            else:
                stats["censored_ranking_positions"] += 1
            for trial in trials:
                if trial["target"] is None:
                    stats["censored_trials"] += 1
                    continue
                ds = [
                    d
                    for d in trial["trajectory"]
                    if d["side"] == "p1"
                    and d["phase"] == "move"
                    and not d["rootIntervention"]
                    and not d.get("rejected")
                ]
                source = result.get("root_provenance", {})
                behavior = "learned-policy" if cfg.get("continuation_net_sha") else "planned"
                kind = (
                    source.get("source", "unknown") + "-root+" + behavior + "-continuation"
                    if trial["action"] == source.get("executed")
                    else "counterfactual-root+" + behavior + "-continuation"
                )
                for d in ds:
                    values[split].append(
                        {
                            "group": bid,
                            "view": d["observation"],
                            "target": d["target"],
                            "weight": 1 / len(ds),
                            "lineage": kind,
                        }
                    )
                stats[kind] += len(ds)
    if (
        groups["train"] & groups["development"]
        or groups["train"] & groups["calibration"]
        or groups["development"] & groups["calibration"]
    ):
        raise ValueError("source battle crosses training/development")
    net_shas = {
        json.loads((out / split / "manifest.json").read_text())["contract"].get("continuation_net_sha")
        for split in groups
        if (out / split / "manifest.json").exists()
    }
    if len(net_shas) != 1:
        raise ValueError("continuation actor differs between splits")
    return ranks, values, dict(stats)


def encode_ranks(rows):
    from pokesim.learn.dataset import choices_of
    from pokesim.learn.features import Features

    features, encoded = Features.load(), []
    for row in rows:
        pairs = choices_of(row["view"], row["actions"])
        if len(pairs) != len(row["actions"]):
            raise ValueError("ranking action missing from its observation")
        state, actions = features.vectors(row["view"], pairs)
        encoded.append(dict(row, state=state, features=actions))
    return encoded


def paired_interval(delta, groups, seed=0, weights=None):
    """Bootstrap source battles, not the strongly dependent branches or turns."""
    unique = sorted(set(groups))
    weights = np.ones(len(delta)) if weights is None else weights
    means = np.asarray(
        [
            np.average(
                [v for v, g in zip(delta, groups) if g == k], weights=[w for w, g in zip(weights, groups) if g == k]
            )
            for k in unique
        ]
    )
    if len(means) < 2:
        return [-1.0, 1.0]
    rng = np.random.default_rng(seed)
    return np.quantile(rng.choice(means, (2000, len(means))).mean(axis=1), [0.05, 0.95]).tolist()


def train(args):
    import torch
    from torch import nn

    from pokesim.learn.features import DEX, SCHEMA, VERSION
    from pokesim.learn.model import PolicyNet, export_json, load_checkpoint, save_checkpoint
    from pokesim.learn.train_value import fit as fit_value
    from pokesim.runtime.bundle import file_hash
    from pokesim.value.contracts import code_hash
    from pokesim.value.contracts import encoder as load_encoder
    from pokesim.value.model import create_model, model_from_config, predict
    from pokesim.value.runtime import ValueEnsemble

    if args.out.exists():
        raise ValueError("choose a new run directory")
    ranks, values, counts = examples(args.data)
    for split in ("train", "development"):
        if not ranks[split] or not values[split]:
            raise ValueError(f"no complete ranking/continuation-value examples in {split}")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    encoded = {k: encode_ranks(v) for k, v in ranks.items()}
    if not args.fixed_policy and not any(np.asarray(r['pair_weights']).any() for r in encoded['train']):
        raise ValueError('no confident paired action preferences; generate more worlds instead of teaching noise')
    source_contracts = [json.loads((args.data / split / 'manifest.json').read_text())['contract']
                        for split in ranks if (args.data / split / 'manifest.json').exists()]
    if len({tuple(c['continuations']) for c in source_contracts}) != 1:
        raise ValueError('opponent continuation league differs between splits')
    continuations = source_contracts[0]['continuations']
    versions = {r['policy_version'] for rows in ranks.values() for r in rows}
    if len(versions) != 1:
        raise ValueError('continuation policy version differs between splits')
    policy_version = next(iter(versions))
    first = encoded["train"][0]
    policy = PolicyNet(
        len(first["state"]), first["features"].shape[1], hidden=64, embed=32, action_hidden=24, scorer_hidden=32
    )
    if args.fixed_policy:
        policy, _ = load_checkpoint(args.fixed_policy.with_suffix(".pt"))
        for split in ("train", "development", "calibration"):
            manifest_path = args.data / split / "manifest.json"
            if manifest_path.exists():
                contract = json.loads(manifest_path.read_text())["contract"]
                if contract.get("continuation_net_sha") != file_sha(args.fixed_policy):
                    raise ValueError("value refresh must follow the fixed ranking policy in every split")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001, weight_decay=0.01)

    def logits(row):
        return policy(
            torch.tensor(row["state"][None]),
            torch.tensor(row["features"][None]),
            torch.ones(1, len(row["actions"]), dtype=torch.bool),
        )[0]

    def rank_metrics(rows, model=True):
        regrets, groups = [], []
        with torch.no_grad():
            for row in rows:
                if model:
                    index = int(logits(row).argmax())
                else:
                    # Compare with the actual plan policy's public choice, recorded before the root intervention.
                    scores = row.get("baseline_action")
                    index = row["actions"].index(scores) if scores in row["actions"] else 0
                q = np.array(row["targets"])
                regrets.append(float(q.max() - q[index]))
                groups.append(row["group"])
        mean = np.mean([np.mean([v for v, g in zip(regrets, groups) if g == k]) for k in set(groups)])
        return {"regret": float(mean), "positions": len(rows), "battles": len(set(groups))}, regrets, groups

    best, saved, best_epoch = rank_metrics(encoded["development"])[0]["regret"], None, 0
    saved = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    battle_counts = Counter(r["group"] for r in encoded["train"])
    for epoch in range(0 if args.fixed_policy else args.epochs):
        for row in encoded["train"]:
            score = logits(row)
            delta = torch.tensor(row['pair_weights'])
            # Only complete, paired and world-robust differences become teachers.
            mask = delta > 0
            if not mask.any():
                continue
            gap = (score[:, None] - score[None, :])[mask]
            loss = (nn.functional.softplus(-gap) * delta[mask]).sum() / mask.sum() / battle_counts[row["group"]]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
        metric = rank_metrics(encoded["development"])[0]["regret"]
        if metric < best - 1e-8:
            best, best_epoch = metric, epoch + 1
            saved = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(saved)
    feature_version = getattr(args, 'feature_version', 'observed-value-v1')
    encoder = load_encoder(feature_version, DEX)
    schema = None

    def value_data(rows):
        nonlocal schema
        items = [encoder.encode(r["view"]) for r in rows]
        schema = items[0][4]
        for item in items:
            if item[4] != schema:
                raise ValueError("value feature schema differs")
        groups = np.array([r["group"] for r in rows])
        weights = np.array([r["weight"] for r in rows], np.float32)
        totals = {g: weights[groups == g].sum() for g in set(groups)}
        weights /= np.array([totals[g] for g in groups])
        return {
            "public": np.stack([i[0] for i in items]),
            "private": np.stack([i[1] for i in items]),
            "baseline": np.stack([i[2] for i in items]),
            "target": np.array([r["target"] for r in rows], np.float32),
            "weight": weights / weights.mean(),
            "group": groups,
        }

    train_data, dev_data = value_data(values["train"]), value_data(values["development"])
    initial = ValueEnsemble(args.init_value, allow_candidate=True) if args.init_value else None
    if initial:
        if initial.features != schema:
            raise ValueError('initial value encoder differs from branch observations')
        value = initial.models[0]
    else:
        value = create_model(schema, getattr(args, 'hidden', 48))
    reference_state = {k: v.detach().clone() for k, v in value.state_dict().items()}
    before = predict(value, dev_data)
    selection = fit_value(
        value,
        train_data,
        dev_data,
        epochs=args.epochs,
        lr=0.0003,
        patience=8,
        batch=256,
        seed=args.seed,
        device="cpu",
        log=lambda s: print(s, flush=True),
    )
    args.out.mkdir(parents=True)
    policy_contract = {
        "head": "ours",
        "feature_version": VERSION,
        "feature_contract": file_hash(SCHEMA),
        "target": "terminal-branch-pairwise-ranking",
        "continuation": continuations,
        "teacher": "complete paired-world CI90 positive; no opponent-policy sign reversal; minimum 4 worlds",
        "source_split_sha": file_sha(args.data / "split.json"),
    }
    save_checkpoint(args.out / "ours.pt", policy, policy_contract)
    export_json(args.out / "ours.json", policy, policy_contract)
    cfg = {
        "target": "terminal_branch_continuation",
        "feature_version": feature_version,
        "feature_code_sha": code_hash(feature_version),
        "dex_sha": file_hash(DEX),
        "features": schema,
        "reference": {"own": "planned", "opponent_pool": continuations, "version": policy_version},
        "source_split_sha": file_sha(args.data / "split.json"),
        "worker_code_sha": tree_sha(ROOT / "services/sim-tools", {".js"}),
        "initial_value_report": file_sha(args.init_value / "report.json") if initial else None,
    }
    if args.fixed_policy:
        # Copy the exact exported actor to preserve its content hash, rather than reserializing it.
        import shutil

        shutil.copyfile(args.fixed_policy, args.out / "ours.json")
        cfg["reference"]["policy_sha"] = file_sha(args.fixed_policy)
    checkpoint = args.out / "value.pt"
    torch.save(
        {
            "schema_version": 1,
            "stage": "terminal_branch_continuation",
            "features": schema,
            "model_config": value.config,
            "state_dict": value.state_dict(),
            "reference": cfg["reference"],
        },
        checkpoint,
    )
    after = predict(value, dev_data)
    report = {
        "provenance": cfg,
        "rank_contract": policy_contract,
        "stages": {"branch": {"checkpoint": checkpoint.name, "sha": file_hash(checkpoint)}},
        "policy": {"path": "ours.json", "sha": file_hash(args.out / "ours.json")},
        "rank": {"epoch": best_epoch, "development": rank_metrics(encoded["development"])[0]},
        "value": {
            "selection": selection,
            "development_brier_before": float(
                np.average((before - dev_data["target"]) ** 2, weights=dev_data["weight"])
            ),
            "development_brier_after": float(np.average((after - dev_data["target"]) ** 2, weights=dev_data["weight"])),
        },
        "counts": counts,
        "data": str(args.data.resolve()),
        "gate": {
            "passed": False,
            "reason": "no untouched calibration cohort; development fit is not adoption evidence",
        },
    }
    if encoded["calibration"] and values["calibration"]:
        # Calibration is scored once, after epoch selection. No threshold/epoch choices use it.
        held = value_data(values["calibration"])
        new = predict(value, held)
        reference = model_from_config(value.config)
        reference.load_state_dict(reference_state)
        old = predict(reference, held)
        delta = (new - held["target"]) ** 2 - (old - held["target"]) ** 2
        interval = paired_interval(delta, held["group"], weights=held["weight"])
        rank_score, regret, groups = rank_metrics(encoded["calibration"])
        _, baseline, _ = rank_metrics(encoded["calibration"], model=False)
        rank_interval = paired_interval(np.array(regret) - baseline, groups)
        enough = len(set(groups)) >= 20
        passed = (
            enough and interval[1] < 0 and np.average(delta, weights=held["weight"]) <= -0.015 and rank_interval[1] < 0
        )
        report["gate"] = {
            "passed": bool(passed),
            "calibration_rank": rank_score,
            "value_brier_delta_interval90": interval,
            "rank_regret_delta_interval90": rank_interval,
            "minimum_battles": 20,
            "calibration_battles": len(set(groups)),
            "reason": "paired battle-level calibration; excludes development and censored branches",
        }
    atomic_json(args.out / "report.json", report)
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--records", type=Path, default=ROOT / "data/raw/ladder")
    gen.add_argument("--team", type=Path, default=ROOT / "data/my_teams/baton.json")
    gen.add_argument("--priors", type=Path, default=ROOT / "data/processed/scoped/priors.json")
    gen.add_argument("--positions", type=Path, help="[{file,turn,split,actions?}]; otherwise mine failures/corrections")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--limit", type=int, default=16, help="battles per split when mining automatically")
    gen.add_argument("--worlds", type=int, default=4)
    gen.add_argument("--seeds", type=int, default=4)
    gen.add_argument("--seed", type=int, default=20261018)
    gen.add_argument("--continuation-net", type=Path, help="fixed own argmax policy for on-policy value refresh")
    gen.add_argument("--max-turns", type=int, default=200)
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--init-value", type=Path)
    fit.add_argument("--fixed-policy", type=Path, help="keep this ranking policy while refreshing its own value")
    fit.add_argument("--epochs", type=int, default=40)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument('--feature-version', choices=['observed-value-v1', 'observed-value-sets-v2'],
                     default='observed-value-v1')
    fit.add_argument('--hidden', type=int, default=48)
    args = parser.parse_args()
    (generate if args.command == "generate" else train)(args)


if __name__ == "__main__":
    main()
