"""Build legal Showdown teams from extracted article sets.

Missing fields are filled from the season's per-species common sets (summarize_sets), then
from simple fallbacks (STAB/high-power learnset moves, a nature matching the attacking side).
Items are kept unique per team (Item Clause). Every team is checked by Showdown's validator.

Usage: uv run python -m pokesim.sim.teams --season M-6
"""
import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select

from pokesim.config import FORMAT, PROCESSED_DIR, ROOT, SHOWDOWN_DIR
from pokesim.data.summarize_sets import SpeciesSummary, summarize
from pokesim.db.models import ArticleSet, BuildArticle, Item, Learnset, Move, Species
from pokesim.db.session import Session

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
PHYSICAL_SPREAD = [2, 32, 0, 0, 0, 32]
SPECIAL_SPREAD = [2, 0, 0, 32, 0, 32]


@dataclass
class BuiltTeam:
    article_id: int
    rank: int | None
    rating: float | None
    player: str | None
    url: str
    sets: list[dict]
    filled: int  # fields filled from defaults (8 per set: 4 moves + item/ability/nature/points)


class TeamBuilder:
    def __init__(self, session, season: str):
        self.s = session
        self.summary: dict[str, SpeciesSummary] = {x.species_id: x for x in summarize(season)}
        self.moves = {m.id: m for m in session.scalars(select(Move))}
        self.items = {i.id: i for i in session.scalars(select(Item))}

    def species(self, species_id: str) -> Species:
        return self.s.get(Species, species_id)

    def learnset(self, species_id: str) -> set[str]:
        return set(self.s.scalars(select(Learnset.move_id).where(Learnset.species_id == species_id)))

    def fill_moves(self, species: Species, moves: list[str]) -> tuple[list[str], int]:
        learnable = self.learnset(species.id)
        out = [m for m in moves if m in learnable]
        filled = 0
        common = self.summary.get(species.id)
        for move, _ in (common.moves.most_common() if common else []):
            if len(out) >= 4:
                break
            if move not in out and move in learnable:
                out.append(move)
                filled += 1
        if len(out) < 4:  # fallback: strongest accurate moves, STAB first
            def power(mid: str) -> float:
                m = self.moves[mid]
                stab = 1.5 if m.type in species.types else 1.0
                return m.base_power * stab * (m.accuracy or 100) / 100
            for move in sorted((m for m in learnable if self.moves[m].base_power > 0 and m not in out),
                               key=power, reverse=True):
                if len(out) >= 4:
                    break
                out.append(move)
                filled += 1
        return out, filled

    def attacking_side(self, moves: list[str]) -> str:
        cats = Counter(self.moves[m].category for m in moves if self.moves[m].category != "Status")
        return "special" if cats["Special"] > cats["Physical"] else "physical"

    def build_set(self, row: ArticleSet, used_items: set[str]) -> tuple[dict, int]:
        species = self.species(row.species_id)
        common = self.summary.get(species.id)
        filled = 0
        moves, n = self.fill_moves(species, list(row.moves))
        filled += n
        side = self.attacking_side(moves)

        item = row.item_id if row.item_id and row.item_id not in used_items else None
        if item is None:
            filled += 1
            item = next((i for i, _ in (common.items.most_common() if common else [])
                         if i not in used_items and not (self.items[i].mega_stone and
                                                         not self._mega_matches(i, species))), None)
        if item:
            used_items.add(item)

        ability = row.ability_id if row.ability_id in species.abilities else None
        if ability is None:
            filled += 1
            ability = next((a for a, _ in (common.abilities.most_common() if common else [])
                            if a in species.abilities), species.abilities[0])

        nature = row.nature
        if nature is None:
            filled += 1
            nature = (common.natures.most_common(1)[0][0] if common and common.natures
                      else ("modest" if side == "special" else "adamant"))

        points = row.stat_points
        if points is None:
            filled += 1
            spread = common.spreads.most_common(1)[0][0] if common and common.spreads else None
            points = [int(x) for x in spread.split("-")] if spread else \
                (SPECIAL_SPREAD if side == "special" else PHYSICAL_SPREAD)

        return {
            "species": species.name,
            "item": self.items[item].name if item else "",
            "ability": ability,
            "nature": nature,
            "moves": moves,
            "evs": dict(zip(STATS, points)),  # Champions: stat points (0-32) are stored in the EV field
            "ivs": dict.fromkeys(STATS, 31),
            "level": 50,
        }, filled

    def _mega_matches(self, item_id: str, species: Species) -> bool:
        mega = self.s.get(Species, self.items[item_id].mega_stone)
        return bool(mega and mega.changes_from == species.name)


def complete_teams(session, season: str) -> list[tuple[BuildArticle, list[ArticleSet]]]:
    """Articles with all 6 sets extracted, one per URL (some articles are listed under two ranks)."""
    articles = session.scalars(select(BuildArticle).where(BuildArticle.season == season)
                               .order_by(BuildArticle.rank)).all()
    seen_urls: set[str] = set()
    out = []
    for art in articles:
        rows = session.scalars(select(ArticleSet).where(ArticleSet.article_id == art.id)
                               .order_by(ArticleSet.slot)).all()
        if len(rows) == 6 and art.url not in seen_urls:
            seen_urls.add(art.url)
            out.append((art, rows))
    return out


def validate(teams: list[list[dict]]) -> list[dict]:
    script = ROOT / "services/sim-tools/validate_teams.js"
    proc = subprocess.run(["node", str(script), str(SHOWDOWN_DIR), FORMAT], input=json.dumps(teams),
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="M-6")
    args = parser.parse_args()

    with Session() as s:
        builder = TeamBuilder(s, args.season)
        built: list[BuiltTeam] = []
        for art, rows in complete_teams(s, args.season):
            used: set[str] = set()
            sets, filled = [], 0
            for row in rows:
                pset, n = builder.build_set(row, used)
                sets.append(pset)
                filled += n
            built.append(BuiltTeam(art.id, art.rank, art.rating, art.player_name, art.url, sets, filled))

    results = validate([t.sets for t in built])
    legal = []
    for team, res in zip(built, results):
        if res["ok"]:
            legal.append({**team.__dict__, "packed": res["packed"]})
        else:
            print(f"rank {team.rank} rejected: {res['problems'][:3]}")
    out = PROCESSED_DIR / f"teams_{args.season}.json"
    out.write_text(json.dumps(legal, ensure_ascii=False, indent=1))
    print(f"{len(legal)}/{len(built)} teams legal -> {out}")


if __name__ == "__main__":
    main()
