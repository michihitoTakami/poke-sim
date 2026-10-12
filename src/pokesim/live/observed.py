"""Opponents' sets seen in our own battles -> data/processed/observed_sets.json, for the set sampler.

From each battle record's protocol log (data/raw/live/logs, records rebuilt from saved frames, Showdown ladder
battles from scripts/ladder.py): per species,
in how many battles it was in the preview, was brought, led; which moves, items, abilities it revealed and whether
it Mega Evolved. pokesim.search.lookahead.SetSampler blends these into the published set data, so moves the
articles never list (Curse Mimikyu, Stealth Rock Archaludon) become possible and frequent ones gain weight.

  uv run python -m pokesim.live.observed            # rebuild from all records
"""
import json
from collections import Counter
from pathlib import Path

from pokesim.config import RAW_DIR, ROOT
from pokesim.rl.myteam import to_id

OUT = ROOT / "data" / "processed" / "observed_sets.json"
SOURCES = [RAW_DIR / "live" / "logs", ROOT / "data" / "processed" / "replayed", RAW_DIR / "ladder"]


def _minutes(rec: dict) -> int:
    started = rec.get("started", "")  # "YYYYMMDD-HHMMSS"
    return int(started[:8]) * 1440 + int(started[9:11]) * 60 + int(started[11:13]) if len(started) >= 13 else 0


def load_records(sources: list[Path] = SOURCES) -> list[dict]:
    """All records, one per battle. The same battle can be in two sources (live record and one rebuilt from frames:
    4+ of the opponent's 6 in common, started within 15 minutes); the later source wins (frames re-read with the
    current reader)."""
    out: list[dict] = []
    for src in sources:
        for p in sorted(src.glob("*.json")):
            rec = json.loads(p.read_text())
            if not (rec.get("preview") and rec.get("log") and rec.get("started", "") >= "20261012-2"):
                continue
            opp = {to_id(s) for s in rec["preview"]["species"]}  # previews can differ by a misread slot
            out = [r for r in out if not (len(opp & {to_id(s) for s in r["preview"]["species"]}) >= 4
                                          and abs(_minutes(r) - _minutes(rec)) <= 15)]
            out.append(rec)
    return out


def collect(records: list[dict], side: str = "p2") -> dict[str, dict]:
    out: dict[str, dict] = {}

    def entry(sid: str) -> dict:
        return out.setdefault(sid, {"previews": 0, "brought": 0, "leads": 0, "moves": Counter(), "items": Counter(),
                                    "abilities": Counter(), "mega": 0})

    for rec in records:
        for sp in rec["preview"]["species"]:
            entry(to_id(sp))["previews"] += 1
        names: dict[str, str] = {}  # "p2a: Charizard" -> species id
        brought: set[str] = set()
        moves: dict[str, set[str]] = {}
        items: dict[str, set[str]] = {}
        abilities: dict[str, set[str]] = {}
        mega: set[str] = set()
        lead = None
        for line in rec["log"]:
            parts = line.split("|")
            if len(parts) < 3 or not parts[2].startswith(side):
                continue
            kind, who = parts[1], parts[2].split(": ", 1)[-1]
            if kind in ("switch", "drag"):
                sid = to_id(parts[3].split(",")[0])
                names[who] = sid
                brought.add(sid)
                lead = lead or sid
            sid = names.get(who)
            if sid is None:
                continue
            if kind == "move" and "[from]" not in line:
                moves.setdefault(sid, set()).add(to_id(parts[3]))
            elif kind in ("-item", "-enditem"):
                items.setdefault(sid, set()).add(to_id(parts[3]))
            elif kind == "-ability":
                abilities.setdefault(sid, set()).add(to_id(parts[3]))
            elif kind == "detailschange" and "-Mega" in parts[3]:
                mega.add(sid)
        for sid in brought:
            e = entry(sid)
            e["brought"] += 1
            e["moves"].update(moves.get(sid, ()))
            e["items"].update(items.get(sid, ()))
            e["abilities"].update(abilities.get(sid, ()))
            e["mega"] += sid in mega
        if lead:
            entry(lead)["leads"] += 1
    return out


def load() -> dict[str, dict]:
    if not OUT.exists():
        return {}
    return {k: {**v, "moves": Counter(v["moves"]), "items": Counter(v["items"]),
                "abilities": Counter(v["abilities"])} for k, v in json.loads(OUT.read_text()).items()}


def main() -> None:
    records = load_records()
    obs = collect(records)
    OUT.write_text(json.dumps({k: {**v, "moves": dict(v["moves"]), "items": dict(v["items"]),
                                   "abilities": dict(v["abilities"])} for k, v in sorted(obs.items())},
                              ensure_ascii=False, indent=1))
    print(f"{len(records)} battles, {sum(1 for v in obs.values() if v['brought'])} species brought -> {OUT}")
    for sid, v in sorted(obs.items(), key=lambda kv: -kv[1]["brought"]):
        if v["brought"]:
            print(f"  {sid:14s} brought {v['brought']}/{v['previews']}  moves {dict(v['moves'])}")


if __name__ == "__main__":
    main()
