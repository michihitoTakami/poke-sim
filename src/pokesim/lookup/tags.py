"""Derive coarse effect tags for abilities/items from Showdown event handlers and descriptions.

Tags feed the UI ("what does this ability do right now?") and the belief model
("which observations can reveal or rule out this ability?").
"""
import re

HANDLER_TAGS: dict[str, str] = {
    "onStart": "switch_in",
    "onSwitchIn": "switch_in",
    "onSwitchOut": "switch_out",
    "onResidual": "end_of_turn",
    "onDamagingHit": "on_hit",
    "onAfterMoveSecondary": "on_hit",
    "onTryHit": "immunity",
    "onImmunity": "immunity",
    "onModifyPriority": "priority",
    "onFractionalPriority": "priority",
    "onModifySpe": "speed",
    "onBasePower": "damage_calc",
    "onModifyAtk": "damage_calc",
    "onModifySpA": "damage_calc",
    "onModifyDef": "damage_calc",
    "onModifySpD": "damage_calc",
    "onModifyDamage": "damage_calc",
    "onSourceModifyDamage": "damage_calc",
    "onSourceModifyAtk": "damage_calc",
    "onSourceModifySpA": "damage_calc",
    "onSourceBasePower": "damage_calc",
    "onEffectiveness": "damage_calc",
    "onModifyType": "type_change",
    "onModifyMove": "move_change",
    "onSetStatus": "status_immunity",
    "onTryBoost": "boost_block",
    "onDamage": "damage_prevention",
    "onTrapPokemon": "trapping",
    "onFoeTrapPokemon": "trapping",
    "onEatItem": "berry",
    "onUpdate": "conditional_trigger",
    "onFaint": "on_faint",
    "onSourceAfterFaint": "on_ko",
    "onWeather": "weather",
    "onPrepareHit": "type_change",
    "onAnyModifyBoost": "damage_calc",
    "onCriticalHit": "crit",
    "onAllyTryHitSide": "immunity",
}

# Abilities implemented outside their own handlers (engine special cases).
MANUAL_TAGS: dict[str, list[str]] = {
    "levitate": ["immunity"],
    "corrosion": ["status_immunity"],
    "dancer": ["move_change"],
    "terashell": ["damage_calc"],
    "earlybird": ["status_immunity"],
}

DESC_TAGS: list[tuple[str, str]] = [
    (r"\b(rain|sun|sunlight|sandstorm|snow|hail)\b", "weather"),
    (r"\bterrain\b", "terrain"),
    (r"\bmega evolv", "mega"),
    (r"\bheal|restores?\b", "recovery"),
    (r"\bcritical hit", "crit"),
    (r"\bcontact\b", "contact"),
    (r"\blowers?\b.*\b(attack|defense|speed)\b", "stat_drop"),
]


def derive_tags(handlers: list[str], text: str, ref_id: str = "") -> list[str]:
    tags = {HANDLER_TAGS[h] for h in handlers if h in HANDLER_TAGS} | set(MANUAL_TAGS.get(ref_id, []))
    lowered = text.lower()
    tags |= {tag for pattern, tag in DESC_TAGS if re.search(pattern, lowered)}
    return sorted(tags)
