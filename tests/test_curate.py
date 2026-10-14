import json

from pokesim.data.curate_battles import curate
from pokesim.data.split_manifest import Splits, make

OPP = ["Garchomp", "Hippowdon", "Rotom-Wash", "Gyarados", "Kingambit", "Primarina"]


def record(started: str, preview=OPP, lead="Garchomp", result="win", **extra) -> dict:
    log = ["|switch|p1a: Scolipede|Scolipede, L50|165/165", f"|switch|p2a: {lead}|{lead}, L50|100/100",
           "|turn|1", "|turn|2"]
    return {"started": started, "result": result, "team": ["Scolipede"], "preview": {"species": list(preview)},
            "lineup": ["Scolipede"], "log": log, **extra}


def write(folder, name, rec):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(rec))


def test_live_and_rebuilt_records_of_one_battle_merge_and_rematches_do_not(tmp_path):
    live, rebuilt, ladder = tmp_path / "logs", tmp_path / "replayed", tmp_path / "ladder"
    write(live, "20261013-021356", record("20261013-021356", result=None))
    misread = OPP[:4] + ["Arboliva", "Arboliva"]  # the rebuilt record read two slots differently
    write(rebuilt, "20261013-021356", record("20261013-021356", preview=misread))
    write(live, "20261013-022000", record("20261013-022000", lead="Gyarados"))  # same team again, 6 min later
    write(ladder, "a", record("20261014-100000", source="showdown battle-x-1", opponent="bob"))
    write(ladder, "b", record("20261014-100500", source="showdown battle-x-2", opponent="bob", result=None))
    battles, _ = curate({"switch_live": live, "switch_rebuilt": rebuilt, "showdown": ladder})
    by_id = {b.battle_id: b for b in battles}
    merged = by_id["switch:20261013-021356"]
    assert [r["kind"] for r in merged.records] == ["switch_rebuilt", "switch_live"]  # the rebuilt one is primary
    assert merged.result == "win"
    rematch = by_id["switch:20261013-022000"]
    assert rematch.merge_pending == ["switch:20261013-021356"]  # close, not confirmed: kept, flagged
    assert by_id["showdown:battle-x-2"].result == "unknown" and "no_result" in by_id["showdown:battle-x-2"].quality
    assert by_id["showdown:battle-x-1"].opponent_group == "showdown:bob"
    assert len(battles) == 4


def test_split_keeps_groups_together_in_time_order(tmp_path):
    ladder = tmp_path / "ladder"
    names = ["ann", "bob", "cat", "dan", "eve", "ann", "fay", "gus", "hal", "ivy"]
    for i, name in enumerate(names):
        write(ladder, f"r{i}", record(f"20261014-1{i:02d}000", source=f"showdown battle-{i}", opponent=name))
    write(ladder, "late", record("20261020-000000", source="showdown battle-late", opponent="zed"))
    battles, _ = curate({"showdown": ladder})
    manifest = make(battles, (0.6, 0.2, 0.2), test_from="20261016-000000")
    split = manifest["battles"]
    assert split["showdown:battle-0"] == "train"
    assert split["showdown:battle-5"] == "train"  # ann's second battle stays with her first (before dev starts)
    assert split["showdown:battle-late"] == "test"  # forward test
    order = ["train", "development", "calibration"]
    starts = [i for i in range(10) if split[f"showdown:battle-{i}"] != "purged"]
    ranks = [order.index(split[f"showdown:battle-{i}"]) for i in starts]
    assert ranks == sorted(ranks)  # no later battle in an earlier split


def test_a_repeat_opponent_after_the_next_split_starts_is_purged(tmp_path):
    ladder = tmp_path / "ladder"
    names = ["ann", "bob", "cat", "dan", "eve", "fay", "gus", "hal", "ivy", "ann"]
    for i, name in enumerate(names):
        write(ladder, f"r{i}", record(f"20261014-1{i:02d}000", source=f"showdown battle-{i}", opponent=name))
    battles, _ = curate({"showdown": ladder})
    manifest = make(battles, (0.6, 0.2, 0.2), test_from="20261016-000000")
    assert manifest["battles"]["showdown:battle-0"] == "train"
    assert manifest["battles"]["showdown:battle-9"] == "purged"


def test_record_keys_follow_the_battle(tmp_path):
    live, rebuilt = tmp_path / "logs", tmp_path / "replayed"
    write(live, "20261013-021356", record("20261013-021356"))
    write(rebuilt, "20261013-021356", record("20261013-021356"))
    battles, _ = curate({"switch_live": live, "switch_rebuilt": rebuilt})
    path = tmp_path / "split.json"
    path.write_text(json.dumps(make(battles, test_from="20261016-000000")))
    assert Splits(path).record_keys(battles, "train") == {"logs/20261013-021356", "replayed/20261013-021356"}
