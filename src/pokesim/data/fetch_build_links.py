"""Collect links to top-ranked players' team articles from GameWith's ranking listing.

Only the article links (season, rank, rating, player name, URL) are stored; the
listing's own team data is not kept. Source page: https://gamewith.jp/pokemon-champions/560474

Usage: uv run python -m pokesim.data.fetch_build_links [--mode single] [--seasons M-5 M-6]
"""
import argparse
import csv

import httpx
from sqlalchemy.dialects.postgresql import insert

from pokesim.config import RAW_DIR
from pokesim.db.models import BuildArticle
from pokesim.db.session import Session

BASE = "https://gamewith-tool.s3.ap-northeast-1.amazonaws.com/pokemon-champions"
HEADERS = {"User-Agent": "Mozilla/5.0 (poke-sim personal research)"}


def available_seasons(client: httpx.Client, mode: str, max_season: int = 30) -> list[str]:
    seasons = []
    for n in range(1, max_season + 1):
        if client.get(f"{BASE}/structure-data-M-{n}-{mode}-update.json").status_code != 200:
            break
        seasons.append(f"M-{n}")
    return seasons


def fetch_links(client: httpx.Client, season: str, mode: str) -> list[dict]:
    name = f"structure-data-{season}-{mode}"
    version = client.get(f"{BASE}/{name}-update.json").raise_for_status().json()["lastUpdate"]
    entries = client.get(f"{BASE}/{name}.json", params={"v": version}).raise_for_status().json()
    return [{"season": season, "battle_mode": mode, "rank": int(e["r"]) if e.get("r") else None,
             "rating": float(e["p"]) if e.get("p") else None, "player_name": e.get("un"), "url": e["url"]}
            for e in entries if e.get("url")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="single", choices=["single", "double"])
    parser.add_argument("--seasons", nargs="*", help="default: all available seasons")
    args = parser.parse_args()

    with httpx.Client(headers=HEADERS, timeout=30) as client:
        seasons = args.seasons or available_seasons(client, args.mode)
        links = [link for s in seasons for link in fetch_links(client, s, args.mode)]

    with Session.begin() as s:
        for link in links:
            s.execute(insert(BuildArticle).values(source="gamewith", **link).on_conflict_do_nothing())

    out = RAW_DIR / f"build_links_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["season", "battle_mode", "rank", "rating", "player_name", "url"])
        writer.writeheader()
        writer.writerows(sorted(links, key=lambda x: (x["season"], x["rank"] or 10**9)))
    print(f"{len(links)} links from {', '.join(seasons)} -> {out}")


if __name__ == "__main__":
    main()
