"""Aggregate article sets into per-species "common set" candidates (the prior for set inference).

Usage: uv run python -m pokesim.data.summarize_sets --season M-6 [--top 15]
"""
import argparse
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import select

from pokesim.db.models import ArticleSet, BuildArticle, NameI18n
from pokesim.db.session import Session


@dataclass
class SpeciesSummary:
    species_id: str
    teams: int = 0
    items: Counter = field(default_factory=Counter)
    abilities: Counter = field(default_factory=Counter)
    natures: Counter = field(default_factory=Counter)
    moves: Counter = field(default_factory=Counter)
    spreads: Counter = field(default_factory=Counter)


def summarize(season: str) -> list[SpeciesSummary]:
    with Session() as s:
        rows = s.scalars(select(ArticleSet).join(BuildArticle).where(BuildArticle.season == season)).all()
    out: dict[str, SpeciesSummary] = {}
    for r in rows:
        sm = out.setdefault(r.species_id, SpeciesSummary(r.species_id))
        sm.teams += 1
        for counter, value in ((sm.items, r.item_id), (sm.abilities, r.ability_id), (sm.natures, r.nature)):
            if value:
                counter[value] += 1
        sm.moves.update(r.moves)
        if r.stat_points:
            sm.spreads["-".join(map(str, r.stat_points))] += 1
    return sorted(out.values(), key=lambda x: -x.teams)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="M-6")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    with Session() as s:
        ja = dict(s.execute(select(NameI18n.ref_id, NameI18n.name).where(NameI18n.lang == "ja")).all())

    def fmt(counter: Counter, n: int = 3) -> str:
        total = sum(counter.values())
        return " / ".join(f"{ja.get(k, k)} {v}/{total}" for k, v in counter.most_common(n)) or "-"

    for sm in summarize(args.season)[:args.top]:
        print(f"■ {ja.get(sm.species_id, sm.species_id)} ({sm.teams} 構築)")
        print(f"   持ち物: {fmt(sm.items)}")
        print(f"   特性  : {fmt(sm.abilities, 2)}")
        print(f"   性格  : {fmt(sm.natures)}")
        print(f"   技    : {fmt(sm.moves, 6)}")
        print(f"   配分  : {fmt(sm.spreads, 2)}")


if __name__ == "__main__":
    main()
