"""Ability/move/item lookup: what an effect does and how it matters in battle."""
from dataclasses import dataclass, field

from sqlalchemy import select

from pokesim.db.models import Ability, Item, Move, NameI18n, Species
from pokesim.db.session import Session
from pokesim.lookup.names import default_index

TAG_LABELS_JA = {
    "switch_in": "登場時に発動", "switch_out": "交代時に発動", "end_of_turn": "ターン終了時",
    "on_hit": "攻撃を受けた時", "immunity": "無効化", "priority": "優先度に影響", "speed": "素早さに影響",
    "damage_calc": "ダメージ計算に影響", "type_change": "タイプ変化", "move_change": "技の性質変化",
    "status_immunity": "状態異常に関係", "boost_block": "能力ダウン無効", "damage_prevention": "ダメージ軽減/無効",
    "trapping": "交代封じ", "berry": "きのみ関連", "conditional_trigger": "条件で発動", "on_faint": "ひんし時",
    "on_ko": "相手を倒した時", "weather": "天候", "terrain": "フィールド", "mega": "メガシンカ",
    "recovery": "回復", "crit": "急所", "contact": "接触", "stat_drop": "能力ダウン",
}


@dataclass
class EffectInfo:
    kind: str
    id: str
    name_en: str
    name_ja: str | None
    short_desc: str
    description_ja: str | None
    tags: list[str] = field(default_factory=list)
    tag_labels_ja: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


_MODELS = {"ability": Ability, "move": Move, "item": Item}


def lookup(text: str, kind: str) -> EffectInfo | None:
    """Look up an ability/move/item by Japanese or English name (fuzzy)."""
    ref_id = default_index().resolve(text, kind)
    if ref_id is None:
        return None
    model = _MODELS[kind]
    with Session() as s:
        row = s.get(model, ref_id)
        ja = s.execute(select(NameI18n).where(NameI18n.kind == kind, NameI18n.ref_id == ref_id,
                                              NameI18n.lang == "ja")).scalar_one_or_none()
        tags = list(getattr(row, "tags", []) or [])
        extra: dict = {}
        if kind == "move":
            extra = {"type": row.type, "category": row.category, "base_power": row.base_power,
                     "accuracy": row.accuracy, "priority": row.priority, "effects": row.effects}
        elif kind == "item" and row.mega_stone:
            extra = {"mega_stone": row.mega_stone}
        elif kind == "ability":
            rows = s.execute(select(Species.id, Species.abilities))
            extra = {"holders": [sid for sid, abilities in rows if ref_id in abilities]}
        return EffectInfo(kind=kind, id=row.id, name_en=row.name, name_ja=ja.name if ja else None,
                          short_desc=row.short_desc, description_ja=ja.description if ja else None, tags=tags,
                          tag_labels_ja=[TAG_LABELS_JA.get(t, t) for t in tags], extra=extra)


def species_abilities(species_text: str) -> list[EffectInfo]:
    """All possible abilities of a species (for 'which ability could this be?' displays)."""
    ref_id = default_index().resolve(species_text, "species")
    if ref_id is None:
        return []
    with Session() as s:
        sp = s.get(Species, ref_id)
    return [info for a in sp.abilities if (info := lookup(a, "ability"))]
