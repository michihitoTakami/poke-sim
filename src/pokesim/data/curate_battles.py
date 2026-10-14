"""One ledger of our battles against people: data/curated/battles.jsonl, decisions.jsonl and manifest.json.

Sources (each record: opponent preview, our lineup, protocol log, result; us = p1):
  switch_live     data/raw/live/logs            read live by the server
  switch_rebuilt  data/processed/replayed       rebuilt from saved frames (scripts/replay_frames.py --records)
  showdown        data/raw/ladder               Showdown ladder (scripts/ladder.py)

A Switch battle can be in both Switch sources. Two records are the same battle only when the opponent's preview
has 5+ species in common, they started within 15 minutes and the first Pokémon each side sent in match; the
rebuilt one is then the primary record. A live record and a rebuilt one with the very same start time (the rebuilt
record keeps it) need only 4 in common: the preview reader misreads a slot or two. Records that are close but do
not pass that check are kept as separate battles and listed in each other's `merge_pending` (a rematch with the
same team is a different battle: we do not drop it on a guess). Showdown battles are identified by their battle tag.

A result that is not win / loss stays "unknown": it is not a draw, and value / win-rate losses must mask it.
`opponent_kind` is "unknown" until there is evidence (a confirmed bot or person); losing to someone is not one.

  uv run python -m pokesim.data.curate_battles        # rebuild the ledger
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pokesim.config import PROCESSED_DIR, RAW_DIR, ROOT
from pokesim.rl.myteam import to_id

CURATED = ROOT / "data" / "curated"
SOURCES = {"switch_live": RAW_DIR / "live" / "logs", "switch_rebuilt": PROCESSED_DIR / "replayed",
           "showdown": RAW_DIR / "ladder"}
SAME_BATTLE_MINUTES = 15
LEDGER_VERSION = 1


@dataclass
class Battle:
    battle_id: str
    source: str  # switch / showdown
    started: str  # YYYYMMDD-HHMMSS
    result: str  # win / loss / unknown
    records: list[dict]  # [{kind, key, path, sha}], primary first; key as in pokesim.rl.human ("replayed/<stem>")
    team_hash: str | None
    lineup: list[str] | None
    opponent_preview: list[str]
    opponent: str | None = None  # opponent's name (Showdown)
    opponent_group: str | None = None  # battles against the same opponent go to the same split
    opponent_kind: str = "unknown"  # human_confirmed / bot_confirmed / unknown
    rating: float | None = None
    policy: str | None = None
    versions: dict | None = None
    turns: int = 0
    quality: list[str] = field(default_factory=list)  # e.g. no_result, short_log, no_lineup
    merge_pending: list[str] = field(default_factory=list)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def stamp_time(stamp: str) -> datetime | None:
    try:
        return datetime.strptime(stamp[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def first_switches(log: list[str]) -> tuple[str | None, str | None]:
    out = {}
    for line in log:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "switch" and parts[2][:2] not in out:
            out[parts[2][:2]] = to_id(parts[3].split(",")[0])
        if len(out) == 2:
            break
    return out.get("p1"), out.get("p2")


def team_hash(team: list | None) -> str | None:
    if not team:
        return None
    return hashlib.sha256(",".join(sorted(to_id(s) for s in team)).encode()).hexdigest()[:12]


def entry(kind: str, path: Path, rec: dict) -> Battle:
    log = rec.get("log") or []
    result = rec.get("result") if rec.get("result") in ("win", "loss") else "unknown"
    started = rec.get("started") or path.stem[:15]
    source = "showdown" if kind == "showdown" else "switch"
    tag = (rec.get("source") or "").removeprefix("showdown ").strip()
    battle_id = f"showdown:{tag}" if source == "showdown" and tag else f"{source}:{started}"
    opponent = rec.get("opponent") or None
    quality = []
    if result == "unknown":
        quality.append("no_result")
    turns = sum(line.startswith("|turn|") for line in log)
    if turns < 2:
        quality.append("short_log")
    if not rec.get("lineup"):
        quality.append("no_lineup")
    preview = (rec.get("preview") or {}).get("species") or []
    return Battle(battle_id=battle_id, source=source, started=started, result=result,
                  records=[{"kind": kind, "key": f"{path.parent.name}/{path.stem}",
                            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                            "sha": sha(path)}],
                  team_hash=team_hash(rec.get("team")), lineup=rec.get("lineup"), opponent_preview=preview,
                  opponent=opponent, opponent_group=f"showdown:{opponent}" if source == "showdown" and opponent
                  else None, rating=rec.get("rating"), policy=rec.get("policy"), versions=rec.get("versions"),
                  turns=turns, quality=quality)


def same_battle(a: tuple[Battle, dict], b: tuple[Battle, dict]) -> str | None:
    """"same" (confirmed duplicate), "pending" (close, not confirmed) or None, for two Switch records."""
    (ba, ra), (bb, rb) = a, b
    ta, tb = stamp_time(ba.started), stamp_time(bb.started)
    if ta is None or tb is None or abs((ta - tb).total_seconds()) > SAME_BATTLE_MINUTES * 60:
        return None
    common = len({to_id(s) for s in ba.opponent_preview} & {to_id(s) for s in bb.opponent_preview})
    if common < 4:
        return None
    leads = first_switches(ra.get("log") or [])
    same_start = ba.started == bb.started and ba.records[0]["kind"] != bb.records[0]["kind"]
    if (common >= 5 or same_start) and None not in leads and leads == first_switches(rb.get("log") or []):
        return "same"
    return "pending"


def curate(sources: dict[str, Path] = SOURCES) -> tuple[list[Battle], list[dict]]:
    """The battles (one per battle, sorted by start) and per-decision rows from the records' saved advice."""
    loaded: list[tuple[Battle, dict]] = []
    for kind, folder in sources.items():
        for path in sorted(folder.glob("*.json")):
            rec = json.loads(path.read_text())
            if not rec.get("log"):
                continue
            loaded.append((entry(kind, path, rec), rec))
    battles: dict[str, tuple[Battle, dict]] = {}
    order = {"switch_rebuilt": 0, "switch_live": 1, "showdown": 0}
    loaded.sort(key=lambda x: (order[x[0].records[0]["kind"]], x[0].started))  # rebuilt records first
    for b, rec in loaded:
        if b.source != "showdown":
            continue
        if b.battle_id in battles:  # the same battle tag twice: one battle, both files listed
            battles[b.battle_id][0].records += b.records
        else:
            battles[b.battle_id] = (b, rec)
    switch = [x for x in loaded if x[0].source == "switch"]
    group = list(range(len(switch)))  # union-find over confirmed duplicates; the root is the primary record

    def root(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    pending = []
    for i in range(len(switch)):
        for j in range(i + 1, len(switch)):
            verdict = same_battle(switch[i], switch[j])
            if verdict == "same":
                group[max(root(i), root(j))] = min(root(i), root(j))
            elif verdict == "pending":
                pending.append((i, j))
    for i, (b, rec) in enumerate(switch):
        r = root(i)
        if r != i:
            switch[r][0].records += b.records
    for i, (b, rec) in enumerate(switch):
        if root(i) == i:
            while b.battle_id in battles:  # two Switch battles started in the same second
                b.battle_id += "+"
            battles[b.battle_id] = (b, rec)
    for i, j in pending:
        a, c = switch[root(i)][0], switch[root(j)][0]
        if a is not c:
            a.merge_pending.append(c.battle_id)
            c.merge_pending.append(a.battle_id)
    for b, _ in battles.values():
        b.merge_pending = sorted(set(b.merge_pending))
    out = sorted((b for b, _ in battles.values()), key=lambda b: (b.started, b.battle_id))
    decisions = []
    for b in out:
        rec = battles[b.battle_id][1]
        for i, adv in enumerate(rec.get("advice") or []):
            decisions.append({"decision_id": f"{b.battle_id}#{i}", "battle_id": b.battle_id, **adv})
    return out, decisions


def write(battles: list[Battle], decisions: list[dict], out: Path = CURATED) -> Path:
    from pokesim.runtime.bundle import git_commit

    out.mkdir(parents=True, exist_ok=True)
    (out / "battles.jsonl").write_text("".join(json.dumps(asdict(b), ensure_ascii=False) + "\n" for b in battles))
    (out / "decisions.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions))
    manifest = {"version": LEDGER_VERSION, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_commit(),
                "battles": len(battles), "decisions": len(decisions),
                "by_source": {s: sum(b.source == s for b in battles) for s in ("switch", "showdown")},
                "results": {r: sum(b.result == r for b in battles) for r in ("win", "loss", "unknown")},
                "merge_pending": sum(bool(b.merge_pending) for b in battles),
                "records": {b.battle_id: [r["sha"] for r in b.records] for b in battles}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return out


def load(path: Path = CURATED / "battles.jsonl") -> list[Battle]:
    return [Battle(**json.loads(line)) for line in path.read_text().splitlines()]


def main() -> None:
    battles, decisions = curate()
    out = write(battles, decisions)
    files = sum(len(b.records) for b in battles)
    print(f"{files} records -> {len(battles)} battles, {len(decisions)} saved decisions -> {out}")
    for s in ("switch", "showdown"):
        sub = [b for b in battles if b.source == s]
        print(f"  {s:8s} {len(sub):4d} battles: " + ", ".join(f"{r} {sum(b.result == r for b in sub)}"
                                                           for r in ("win", "loss", "unknown")))
    pending = [b for b in battles if b.merge_pending]
    if pending:
        print(f"  {len(pending)} battles with an unconfirmed possible duplicate (merge_pending):")
        for b in pending:
            print(f"    {b.battle_id} ~ {', '.join(b.merge_pending)}")


if __name__ == "__main__":
    main()
