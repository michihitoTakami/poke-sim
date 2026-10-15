"""Export only observed, non-intervened p1 decisions with actual terminal outcomes from a frozen source split."""
import json
from collections import Counter
from pathlib import Path

from pokesim.evaluation.branch_matches import atomic_json, digest, file_sha, load_result


def export_value(out: Path, dataset: Path) -> dict:
    out, dataset = Path(out), Path(dataset)
    manifest = json.loads((out / "manifest.json").read_text())
    cfg = manifest["contract"]
    if (manifest["experiment_id"] != digest(cfg) or not manifest.get("training_ready") or
            manifest.get("information_scope") != "public_observation" or
            cfg.get("split") not in {"train", "development"} or
            cfg.get("source_scope", {}).get("split") != cfg["split"]):
        raise ValueError("value export requires a scoped public-observation experiment")
    mine = {c.split('/')[0] for c in cfg["continuations"]}
    if len(mine) != 1:
        raise ValueError("a value dataset requires one fixed own continuation policy")
    metadata_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    if dataset.exists() or metadata_path.exists():
        raise ValueError("refusing to overwrite an existing dataset or its metadata")
    dataset.parent.mkdir(parents=True, exist_ok=True)
    temp = dataset.with_suffix(dataset.suffix + ".tmp")
    if temp.exists():
        raise ValueError("unfinished dataset temp file exists; choose a new --dataset")
    counts, groups, source_files, n = Counter(), set(), {}, 0
    try:
        with temp.open('x') as target:
            for case in cfg["positions"]:
                path = out / "positions" / f"{case['id']}.json"
                if not path.exists():
                    counts["positions_not_generated"] += 1
                    continue
                row = load_result(path, manifest, case)
                result = row["result"]
                if result.get("informationScope") != "public_observation" or not result.get("policyVersion"):
                    raise ValueError("result is not from the public reference policy")
                bid = cfg["battle_ids"][case["file"]]
                if bid not in cfg["source_scope"]["records"]:
                    raise ValueError("source battle is not a member of the frozen split")
                source_files[case["id"]] = row["payload_sha"]
                for trial in result["trials"]:
                    if trial["target"] is None:
                        counts["censored_or_failed_trials"] += 1
                        continue
                    eligible = [d for d in trial["trajectory"] if d["side"] == 'p1' and d["phase"] == 'move'
                                and not d["rootIntervention"] and not d.get('rejected')]
                    counts["excluded_root_or_other_decisions"] += len(trial["trajectory"]) - len(eligible)
                    for d in eligible:
                        sample = {"schema_version": 1, "split": cfg["split"], "group": bid,
                                  "position_id": case["id"], "world": trial["world"], "seed": trial["seed"],
                                  "root_action": trial["action"],
                                  "continuation": trial["continuation"], "step": d["step"],
                                  "reference_policy_version": result["policyVersion"],
                                  "observation": d["observation"], "legal_actions": d["legal"],
                                  "behavior_action": d["action"], "target": d["target"],
                                  "sample_weight": 1 / len(eligible)}
                        target.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + '\n')
                        groups.add(bid)
                        n += 1
        if not n:
            raise ValueError("no non-intervened terminal p1 move decisions to export")
        meta = {"schema_version": 1, "experiment_id": manifest["experiment_id"], "split": cfg["split"],
                "information_scope": "public_observation", "target": "terminal_outcome_under_public_reference",
                "source_scope": cfg["source_scope"], "source_payloads": source_files, "groups": sorted(groups),
                "rows": n, "excluded": dict(counts), "own_reference": next(iter(mine)),
                "opponent_pool": cfg["continuations"], "dataset_sha": file_sha(temp)}
        # Publish metadata before the data: an interruption cannot leave an unversioned training file.
        atomic_json(metadata_path, meta)
        temp.rename(dataset)
        return meta
    finally:
        if temp.exists():
            temp.unlink()
