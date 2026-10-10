"""Download team articles (構築記事) linked from build_articles, politely and with a local cache.

Only hosts whose terms allow automated access are fetched (see ALLOWED_HOSTS).
Pokesol forbids automated collection and note.com serves a bot challenge, so those are
imported manually instead.

Usage: uv run python -m pokesim.data.fetch_articles --season M-6 [--limit 20]
"""
import argparse
import re
import time
from pathlib import Path

import httpx
from sqlalchemy import select

from pokesim.config import RAW_DIR
from pokesim.db.models import BuildArticle
from pokesim.db.session import Session

ALLOWED_HOSTS = re.compile(r"\.(hatenablog\.(com|jp)|hateblo\.jp|hatenadiary\.(org|jp|com)|hatenablog-parts\.com)$")
HEADERS = {"User-Agent": "Mozilla/5.0 (poke-sim personal research)"}
DELAY_SEC = 3.0
CACHE_DIR = RAW_DIR / "articles"


def cache_path(article_id: int) -> Path:
    return CACHE_DIR / f"{article_id}.html"


def is_allowed(url: str) -> bool:
    host = httpx.URL(url).host
    return bool(ALLOWED_HOSTS.search("." + host))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="M-6")
    parser.add_argument("--mode", default="single")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with Session() as s:
        rows = s.execute(select(BuildArticle.id, BuildArticle.url)
                         .where(BuildArticle.season == args.season, BuildArticle.battle_mode == args.mode)
                         .order_by(BuildArticle.rank)).all()
    targets = [(aid, url) for aid, url in rows if is_allowed(url) and not cache_path(aid).exists()]
    if args.limit:
        targets = targets[:args.limit]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(targets)} to fetch ({len(rows)} links in {args.season})", flush=True)

    ok = 0
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for i, (aid, url) in enumerate(targets):
            try:
                resp = client.get(url)
                resp.raise_for_status()
                cache_path(aid).write_text(resp.text, encoding="utf-8")
                ok += 1
            except httpx.HTTPError as e:
                print(f"  failed {url}: {e}", flush=True)
            if i + 1 < len(targets):
                time.sleep(DELAY_SEC)
    print(f"fetched {ok}/{len(targets)}")


if __name__ == "__main__":
    main()
