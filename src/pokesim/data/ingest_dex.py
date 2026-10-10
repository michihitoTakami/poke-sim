"""Load the exported Showdown dex and PokeAPI Japanese names into the database.

Usage: uv run alembic upgrade head && uv run python -m pokesim.data.ingest_dex
Requires data/processed/dex.json (services/dex-export/export.js) and data/raw/pokeapi/*.csv.
"""
import csv
import json
import re
import subprocess
from collections import defaultdict

from sqlalchemy import delete

from pokesim.config import FORMAT, PROCESSED_DIR, RAW_DIR, ROOT, SHOWDOWN_DIR
from pokesim.db.models import Ability, Item, Learnset, Move, NameI18n, Species, TypeEffectiveness
from pokesim.db.session import Session
from pokesim.lookup.tags import derive_tags

# Champions-new entries without Japanese text in Showdown or PokeAPI yet.
JA_NAME_OVERRIDES: dict[tuple[str, str], str] = {
    ("ability", "auraguard"): "はどうのぼうご",  # 波動の防護 (Lucario-Mega-Z)
}

LANG_JA_KANA = "1"  # ja-Hrkt: what the game displays
LANG_EN = "9"


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def read_csv(name: str) -> list[dict]:
    with open(RAW_DIR / "pokeapi" / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def localized(table: str, key: str, ids: dict[str, str]) -> dict[str, dict[str, str]]:
    """Map Showdown id -> {lang: name} for a PokeAPI *_names table keyed by `key`."""
    by_ref: dict[str, dict[str, str]] = defaultdict(dict)
    for row in read_csv(table):
        if row["local_language_id"] in (LANG_JA_KANA, LANG_EN):
            by_ref[row[key]][row["local_language_id"]] = row["name"]
    out = {}
    for ref, names in by_ref.items():
        if (sd_id := ids.get(ref)) is None and LANG_EN in names:
            sd_id = to_id(names[LANG_EN])
        if sd_id:
            out[sd_id] = names
    return out


def ability_descriptions() -> dict[str, str]:
    """Latest Japanese (kana) flavor text per PokeAPI ability id."""
    best: dict[str, tuple[int, str]] = {}
    for row in read_csv("ability_flavor_text"):
        if row["language_id"] == LANG_JA_KANA:
            vg = int(row["version_group_id"])
            if row["ability_id"] not in best or vg > best[row["ability_id"]][0]:
                best[row["ability_id"]] = (vg, row["flavor_text"].replace("\n", ""))
    return {k: v[1] for k, v in best.items()}


def has_japanese(text: str | None) -> bool:
    return bool(text) and bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text))


def species_ja_names(species: list[dict]) -> dict[str, str]:
    base = {}
    for row in read_csv("pokemon_species_names"):
        if row["local_language_id"] == LANG_JA_KANA:
            base[int(row["pokemon_species_id"])] = row["name"]
    form_names = {}
    pokemon_ident = {r["id"]: r["identifier"] for r in read_csv("pokemon")}
    form_to_pokemon = {r["id"]: pokemon_ident.get(r["pokemon_id"], "") for r in read_csv("pokemon_forms")}
    for row in read_csv("pokemon_form_names"):
        if row["local_language_id"] == LANG_JA_KANA and row["pokemon_name"]:
            form_names[to_id(form_to_pokemon.get(row["pokemon_form_id"], ""))] = row["pokemon_name"]
    out = {}
    for s in species:
        if has_japanese(s["nameJa"]):
            out[s["id"]] = s["nameJa"]
        elif s["id"] in form_names:
            out[s["id"]] = form_names[s["id"]]
        elif s["isMega"]:  # Champions-only megas are not in PokeAPI yet
            suffix = s["forme"].removeprefix("Mega").strip("-")
            out[s["id"]] = f"メガ{base.get(s['num'], s['baseSpecies'])}{suffix}"
        elif s["num"] in base:
            out[s["id"]] = base[s["num"]]
    return out


def export_dex() -> dict:
    out = PROCESSED_DIR / "dex.json"
    script = ROOT / "services/dex-export/export.js"
    subprocess.run(["node", str(script), str(SHOWDOWN_DIR), FORMAT, str(out)], check=True)
    return json.loads(out.read_text())


def main() -> None:
    dex = export_dex()
    ability_ids = {a["name"]: a["id"] for a in dex["abilities"]}

    ab_names = localized("ability_names", "ability_id", {})
    ab_desc_by_pokeapi = ability_descriptions()
    ab_pokeapi_id = {to_id(r["identifier"]): r["id"] for r in read_csv("abilities")}
    move_names = localized("move_names", "move_id", {})
    item_names = localized("item_names", "item_id", {})
    type_names = localized("type_names", "type_id", {})
    (PROCESSED_DIR / "messages.json").write_text(json.dumps(dex["messages"], ensure_ascii=False))
    sp_ja = species_ja_names(dex["species"])
    base_ja = {sp["baseSpecies"]: sp_ja[sp["id"]] for sp in dex["species"] if not sp["forme"] and sp["id"] in sp_ja}
    for item in dex["items"]:
        if isinstance(item["megaStone"], dict) and not item["nameJa"] \
                and LANG_JA_KANA not in item_names.get(item["id"], {}):
            # New Champions stones: follow the official "<species>ナイト[Ｘ/Ｙ/Ｚ]" pattern (provisional).
            base_name, mega_name = next(iter(item["megaStone"].items()))
            suffix = {"X": "Ｘ", "Y": "Ｙ", "Z": "Ｚ"}.get(mega_name.rsplit("-", 1)[-1], "")
            if base := base_ja.get(base_name):
                item["nameJa"] = f"{base}ナイト{suffix}"

    with Session.begin() as s:
        for model in (Learnset, NameI18n, TypeEffectiveness, Species, Move, Ability, Item):
            s.execute(delete(model))

        for a in dex["abilities"]:
            s.add(Ability(id=a["id"], name=a["name"], short_desc=a["shortDesc"], desc=a["desc"],
                          rating=a["rating"], tags=derive_tags(a["handlers"], a["shortDesc"], a["id"])))
        for i in dex["items"]:
            mega = to_id(next(iter(i["megaStone"].values()))) if i["megaStone"] else None
            s.add(Item(id=i["id"], name=i["name"], short_desc=i["shortDesc"], mega_stone=mega,
                       is_berry=i["isBerry"], tags=derive_tags(i["handlers"], i["shortDesc"])))
        for m in dex["moves"]:
            effects = {k: m[k] for k in ("secondary", "drain", "recoil", "boosts", "self", "status", "volatileStatus",
                                         "weather", "terrain", "sideCondition", "multihit", "selfSwitch",
                                         "forceSwitch", "critRatio") if m[k]}
            s.add(Move(id=m["id"], name=m["name"], type=m["type"], category=m["category"],
                       base_power=m["basePower"], accuracy=None if m["accuracy"] is True else m["accuracy"],
                       pp=m["pp"], priority=m["priority"], target=m["target"], flags=m["flags"],
                       short_desc=m["shortDesc"], effects=effects))
        s.flush()
        for sp in dex["species"]:
            s.add(Species(id=sp["id"], name=sp["name"], num=sp["num"], base_species=sp["baseSpecies"],
                          forme=sp["forme"], types=sp["types"], base_stats=sp["baseStats"],
                          abilities=[ability_ids[a] for a in sp["abilities"] if a in ability_ids],
                          weightkg=sp["weightkg"], is_mega=sp["isMega"], required_item=sp["requiredItem"],
                          changes_from=sp["changesFrom"]))
        s.flush()
        for sp in dex["species"]:
            s.add_all(Learnset(species_id=sp["id"], move_id=mv) for mv in sp["learnset"])
        for atk, row in dex["typechart"].items():
            s.add_all(TypeEffectiveness(attacking=atk, defending=d, multiplier=v) for d, v in row.items())

        def add_names(kind: str, entries: list[dict], pokeapi: dict[str, dict[str, str]],
                      descs: dict[str, str] | None = None):
            """Japanese name: Showdown's ja text first, PokeAPI (ja-Hrkt) as fallback."""
            for e in entries:
                s.add(NameI18n(kind=kind, ref_id=e["id"], lang="en", name=e["name"]))
                ja = (e.get("nameJa") if has_japanese(e.get("nameJa")) else None) \
                    or pokeapi.get(e["id"], {}).get(LANG_JA_KANA) \
                    or JA_NAME_OVERRIDES.get((kind, e["id"]))
                if ja:
                    s.add(NameI18n(kind=kind, ref_id=e["id"], lang="ja", name=ja,
                                   description=(descs or {}).get(e["id"])))

        ab_desc = {sd: ab_desc_by_pokeapi[pid] for sd, pid in ab_pokeapi_id.items() if pid in ab_desc_by_pokeapi}
        add_names("ability", dex["abilities"], ab_names, ab_desc)
        add_names("move", dex["moves"], move_names)
        add_names("item", dex["items"], item_names)
        add_names("type", [{"id": to_id(t), "name": t, "nameJa": dex["typeNamesJa"].get(t)} for t in dex["typechart"]],
                  type_names)
        add_names("nature", [{"id": to_id(n), "name": n, "nameJa": ja} for n, ja in dex["natureNamesJa"].items()], {})
        for sp in dex["species"]:
            s.add(NameI18n(kind="species", ref_id=sp["id"], lang="en", name=sp["name"]))
            if sp["id"] in sp_ja:
                s.add(NameI18n(kind="species", ref_id=sp["id"], lang="ja", name=sp_ja[sp["id"]]))
        missing = {k: sum(1 for e in v if not (e.get("nameJa") or LANG_JA_KANA in names.get(e["id"], {})))
                   for k, v, names in (("species", dex["species"], {k: {LANG_JA_KANA: v} for k, v in sp_ja.items()}),
                                       ("item", dex["items"], item_names))}
        print("entries without a Japanese name:", missing)

    print(f"ingested {len(dex['species'])} species, {len(dex['moves'])} moves, "
          f"{len(dex['abilities'])} abilities, {len(dex['items'])} items")


if __name__ == "__main__":
    main()
