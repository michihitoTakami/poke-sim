"""Relational schema: dex master data, i18n names, usage snapshots, replays and own battles."""
from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class Base(DeclarativeBase):
    pass


class Species(Base):
    __tablename__ = "species"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Showdown id, e.g. "charizardmegax"
    name: Mapped[str] = mapped_column(String(64))
    num: Mapped[int] = mapped_column(Integer)
    base_species: Mapped[str] = mapped_column(String(64))
    forme: Mapped[str] = mapped_column(String(32), default="")
    types: Mapped[list] = mapped_column(Json)
    base_stats: Mapped[dict] = mapped_column(Json)
    abilities: Mapped[list] = mapped_column(Json)  # ability ids
    weightkg: Mapped[float] = mapped_column(Float)
    is_mega: Mapped[bool]
    required_item: Mapped[str | None] = mapped_column(String(64))
    changes_from: Mapped[str | None] = mapped_column(String(64))
    battle_only: Mapped[str | None] = mapped_column(String(64))  # in-battle forme of this species (Mimikyu-Busted)


class Move(Base):
    __tablename__ = "moves"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(16))
    base_power: Mapped[int]
    accuracy: Mapped[int | None]  # None = never misses
    pp: Mapped[int]
    priority: Mapped[int]
    target: Mapped[str] = mapped_column(String(32))
    flags: Mapped[list] = mapped_column(Json)
    short_desc: Mapped[str] = mapped_column(Text)
    effects: Mapped[dict] = mapped_column(Json)  # secondary/boosts/status/drain/etc.


class Ability(Base):
    __tablename__ = "abilities"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    short_desc: Mapped[str] = mapped_column(Text)
    desc: Mapped[str] = mapped_column(Text)
    rating: Mapped[float]
    tags: Mapped[list] = mapped_column(Json)  # derived effect tags, see pokesim.lookup.tags


class Item(Base):
    __tablename__ = "items"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    short_desc: Mapped[str] = mapped_column(Text)
    mega_stone: Mapped[str | None] = mapped_column(String(64))  # resulting species name
    is_berry: Mapped[bool]
    tags: Mapped[list] = mapped_column(Json)


class Learnset(Base):
    __tablename__ = "learnsets"
    species_id: Mapped[str] = mapped_column(ForeignKey("species.id"), primary_key=True)
    move_id: Mapped[str] = mapped_column(ForeignKey("moves.id"), primary_key=True)


class TypeEffectiveness(Base):
    __tablename__ = "type_chart"
    attacking: Mapped[str] = mapped_column(String(16), primary_key=True)
    defending: Mapped[str] = mapped_column(String(16), primary_key=True)
    multiplier: Mapped[float]


class NameI18n(Base):
    """Localized names, used for OCR matching and UI display."""
    __tablename__ = "names_i18n"
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)  # species/move/ability/item/type/nature
    ref_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    lang: Mapped[str] = mapped_column(String(8), primary_key=True)  # ja / en
    name: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str | None] = mapped_column(Text)


class UsageSnapshot(Base):
    __tablename__ = "usage_snapshots"
    __table_args__ = (UniqueConstraint("source", "season", "format", "species_id", "captured_on"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))  # pokemon-zone / smogon
    season: Mapped[str] = mapped_column(String(32))
    format: Mapped[str] = mapped_column(String(64))
    species_id: Mapped[str] = mapped_column(String(64), index=True)
    rank: Mapped[int | None]
    usage: Mapped[float | None]
    items: Mapped[dict] = mapped_column(Json, default=dict)  # id -> share
    abilities: Mapped[dict] = mapped_column(Json, default=dict)
    moves: Mapped[dict] = mapped_column(Json, default=dict)
    natures: Mapped[dict] = mapped_column(Json, default=dict)
    spreads: Mapped[dict] = mapped_column(Json, default=dict)  # "nature:hp/atk/def/spa/spd/spe" -> share
    teammates: Mapped[dict] = mapped_column(Json, default=dict)
    captured_on: Mapped[datetime] = mapped_column(server_default=func.now())


class Replay(Base):
    __tablename__ = "replays"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    format: Mapped[str] = mapped_column(String(64), index=True)
    uploadtime: Mapped[int]
    rating: Mapped[int | None]
    players: Mapped[list] = mapped_column(Json)
    log: Mapped[str] = mapped_column(Text)


class PreviewRecord(Base):
    """One side of a team preview: the 6 shown, and which were brought (as far as observed)."""
    __tablename__ = "preview_records"
    id: Mapped[int] = mapped_column(primary_key=True)
    replay_id: Mapped[str | None] = mapped_column(ForeignKey("replays.id"), index=True)
    source: Mapped[str] = mapped_column(String(16))  # replay / selfplay / obs
    side_team: Mapped[list] = mapped_column(Json)  # 6 species ids
    opp_team: Mapped[list] = mapped_column(Json)
    brought: Mapped[list] = mapped_column(Json)  # observed species ids, lead first
    brought_complete: Mapped[bool]  # False when fewer than 3 were revealed
    won: Mapped[bool | None]
    rating: Mapped[int | None]


class MyTeam(Base):
    __tablename__ = "my_teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    packed: Mapped[str] = mapped_column(Text)  # Showdown packed team
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class MyBattle(Base):
    __tablename__ = "my_battles"
    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("my_teams.id"))
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    events: Mapped[list] = mapped_column(Json, default=list)  # normalized protocol-like events from vision
    result: Mapped[str | None] = mapped_column(String(8))


class BuildArticle(Base):
    """Link to a top player's team write-up (構築記事), collected from a ranking listing."""
    __tablename__ = "build_articles"
    __table_args__ = (UniqueConstraint("source", "season", "battle_mode", "rank", "url"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))  # gamewith
    season: Mapped[str] = mapped_column(String(16), index=True)  # M-6
    battle_mode: Mapped[str] = mapped_column(String(8))  # single / double
    rank: Mapped[int | None]
    rating: Mapped[float | None]
    player_name: Mapped[str | None] = mapped_column(String(64))
    url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ArticleSet(Base):
    """One Pokémon set extracted from a team article (fields are None when not found)."""
    __tablename__ = "article_sets"
    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("build_articles.id"), index=True)
    slot: Mapped[int]  # order of appearance in the article
    species_id: Mapped[str] = mapped_column(String(64))
    item_id: Mapped[str | None] = mapped_column(String(64))
    ability_id: Mapped[str | None] = mapped_column(String(64))
    nature: Mapped[str | None] = mapped_column(String(16))
    moves: Mapped[list] = mapped_column(Json, default=list)
    stat_points: Mapped[list | None] = mapped_column(Json)  # [hp, atk, def, spa, spd, spe], 0-32 each
    problems: Mapped[list] = mapped_column(Json, default=list)  # legality issues found by validation
    raw_text: Mapped[str] = mapped_column(Text)
    parsed_at: Mapped[datetime] = mapped_column(server_default=func.now())
