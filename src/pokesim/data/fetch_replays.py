"""Download public Showdown replays of the Champions BSS format, highest rated first, into the replays table.

Uses the public replay search API (sorted by rating) and caches each replay to data/raw/replays/{id}.json.

Usage: uv run python -m pokesim.data.fetch_replays [--min-rating 1250] [--max-pages 30]
"""
import argparse
import json
import time

import httpx
from sqlalchemy.dialects.postgresql import insert

from pokesim.config import FORMAT, RAW_DIR
from pokesim.db.models import Replay
from pokesim.db.session import Session

BASE = "https://replay.pokemonshowdown.com"
HEADERS = {"User-Agent": "poke-sim personal research"}
DELAY_SEC = 1.0
CACHE_DIR = RAW_DIR / "replays"


def search(client: httpx.Client, format_id: str, min_rating: int, max_pages: int) -> list[dict]:
    found: dict[str, dict] = {}
    for page in range(1, max_pages + 1):
        rows = client.get(f"{BASE}/search.json", params={"format": format_id, "sort": "rating", "page": page})
        rows = rows.raise_for_status().json()
        found.update((r["id"], r) for r in rows if (r.get("rating") or 0) >= min_rating)
        if len(rows) < 51 or min((r.get("rating") or 0) for r in rows) < min_rating:
            break
        time.sleep(DELAY_SEC)
    return list(found.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default=FORMAT)
    parser.add_argument("--min-rating", type=int, default=1250)
    parser.add_argument("--max-pages", type=int, default=30)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        hits = search(client, args.format, args.min_rating, args.max_pages)
        todo = [h for h in hits if not (CACHE_DIR / f"{h['id']}.json").exists()]
        print(f"{len(hits)} replays rated >= {args.min_rating}, {len(todo)} to download", flush=True)
        for i, h in enumerate(todo):
            try:
                data = client.get(f"{BASE}/{h['id']}.json").raise_for_status().json()
                (CACHE_DIR / f"{h['id']}.json").write_text(json.dumps(data), encoding="utf-8")
            except httpx.HTTPError as e:
                print(f"  failed {h['id']}: {e}", flush=True)
            if i + 1 < len(todo):
                time.sleep(DELAY_SEC)

    with Session.begin() as s:
        for h in hits:
            path = CACHE_DIR / f"{h['id']}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                s.execute(insert(Replay).values(id=h["id"], format=args.format, uploadtime=data["uploadtime"],
                                                rating=h.get("rating"), players=data["players"], log=data["log"])
                          .on_conflict_do_nothing())
    print("stored", len(hits))


if __name__ == "__main__":
    main()
