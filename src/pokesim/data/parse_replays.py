"""Turn stored Showdown replays into partially known teams (one per side) in build_articles / article_sets.

From each log we read the 6 previewed species and whatever the battle revealed: moves used, items (consumed,
knocked off, healing/recoil messages, mega stones), and abilities (announced, or named in [from] tags).
Unrevealed fields stay empty and are filled later by the team builder from usage statistics.

Usage: uv run python -m pokesim.data.parse_replays
"""
import re
from dataclasses import dataclass, field

from sqlalchemy import delete, select

from pokesim.db.models import ArticleSet, BuildArticle, Item, Replay, Species
from pokesim.db.session import Session

SOURCE = "showdown_replay"
SEASON = "SD-M-C"
REPLAY_URL = "https://replay.pokemonshowdown.com/{id}#{side}"


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class MonInfo:
    species: str
    moves: list[str] = field(default_factory=list)
    item: str | None = None
    ability: str | None = None
    mega: bool = False


@dataclass
class SideInfo:
    player: str = ""
    preview: list[str] = field(default_factory=list)
    mons: dict[str, MonInfo] = field(default_factory=dict)  # by base species id
    nick: dict[str, str] = field(default_factory=dict)  # "p1a: Nick" -> base species id
    won: bool = False


def base_species(details: str, species: dict[str, Species]) -> str:
    sid = to_id(details.split(",")[0])
    sp = species.get(sid)
    if sp is not None and sp.battle_only:
        return to_id(sp.battle_only)
    if sp is not None and sp.is_mega and sp.changes_from:
        return to_id(sp.changes_from)
    return sid


def slot_key(ident: str) -> str:
    """'p1a: Nick' -> 'p1: Nick' (the same mon across active-slot letters)."""
    side, _, nick = ident.partition(":")
    return side[:2] + ":" + nick.strip()


def parse_log(log: str, species: dict[str, Species]) -> dict[str, SideInfo]:
    sides = {"p1": SideInfo(), "p2": SideInfo()}
    winner = None

    def mon(ident: str) -> MonInfo | None:
        side = sides.get(ident[:2])
        sid = side.nick.get(slot_key(ident)) if side else None
        return side.mons.get(sid) if sid else None

    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 3:
            continue
        kind = parts[1]
        tags = {p.split(" ", 1)[0]: p.split(" ", 1)[1] for p in parts[3:] if p.startswith("[") and " " in p}
        if kind == "player" and len(parts) > 3 and parts[3]:
            sides[parts[2]].player = parts[3]
        elif kind == "poke":
            sid = base_species(parts[3], species)
            sides[parts[2]].preview.append(sid)
            sides[parts[2]].mons.setdefault(sid, MonInfo(sid))
        elif kind in ("switch", "drag"):
            side = sides[parts[2][:2]]
            sid = base_species(parts[3], species)
            side.nick[slot_key(parts[2])] = sid
            side.mons.setdefault(sid, MonInfo(sid))
        elif kind == "move" and "[from]" not in line and (m := mon(parts[2])):
            mid = to_id(parts[3])
            if mid not in m.moves and mid != "struggle":
                m.moves.append(mid)
        elif kind == "-mega" and (m := mon(parts[2])):
            m.mega, m.item = True, to_id(parts[4]) if len(parts) > 4 else m.item
        elif kind in ("-enditem", "-item") and (m := mon(parts[2])):
            if "move: Trick" not in line and "move: Switcheroo" not in line and "[from] move: Thief" not in line:
                m.item = m.item or to_id(parts[3])
        elif kind == "-ability" and (m := mon(parts[2])) and not m.mega:
            m.ability = m.ability or to_id(parts[3])
        elif kind == "win":
            winner = parts[2]
        for tag, value in tags.items():
            if tag != "[from]" or ":" not in value:
                continue
            what, name = (x.strip() for x in value.split(":", 1))
            owner = mon(tags["[of]"]) if "[of]" in tags else mon(parts[2])
            if owner is None:
                continue
            if what == "item":
                owner.item = owner.item or to_id(name)
            elif what == "ability" and not owner.mega:
                owner.ability = owner.ability or to_id(name)
    for side in sides.values():
        side.won = winner is not None and winner == side.player
    return sides


def main() -> None:
    with Session.begin() as s:
        species = {sp.id: sp for sp in s.scalars(select(Species))}
        items = set(s.scalars(select(Item.id)))
        old = select(BuildArticle.id).where(BuildArticle.source == SOURCE)
        s.execute(delete(ArticleSet).where(ArticleSet.article_id.in_(old)))
        s.execute(delete(BuildArticle).where(BuildArticle.source == SOURCE))
        n_teams = 0
        for rep in s.scalars(select(Replay)):
            for side_id, side in parse_log(rep.log, species).items():
                if len(side.preview) != 6 or any(sid not in species for sid in side.preview):
                    continue
                art = BuildArticle(source=SOURCE, season=SEASON, battle_mode="single", rank=None,
                                   rating=rep.rating, player_name=side.player,
                                   url=REPLAY_URL.format(id=rep.id, side=side_id))
                s.add(art)
                s.flush()
                for slot, sid in enumerate(side.preview, 1):
                    m = side.mons[sid]
                    s.add(ArticleSet(article_id=art.id, slot=slot, species_id=sid,
                                     item_id=m.item if m.item in items else None, ability_id=m.ability,
                                     nature=None, moves=m.moves[:4], stat_points=None, problems=[], raw_text=""))
                n_teams += 1
        print(f"{n_teams} teams from replays")


if __name__ == "__main__":
    main()
