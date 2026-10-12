"""Reference images for recognising Pokémon on the team preview screen.

Downloads the Pokémon HOME render of every Champions-legal species (data/raw/sprites/legal.json, exported from
Showdown) from PokeAPI into data/raw/sprites/home/<showdown id>.png. These are copyrighted game assets: they stay
local (data/raw is git-ignored) and are only used for matching.

Usage: uv run python -m pokesim.vision.refs
"""
import json
import time

import httpx

from pokesim.config import RAW_DIR

SPRITE_DIR = RAW_DIR / "sprites"
HOME_DIR = SPRITE_DIR / "home"
API = "https://pokeapi.co/api/v2"
DELAY = 0.5  # PokeAPI fair use: cache everything, go slowly
# Showdown name -> PokeAPI pokemon name, where the simple lower-case/hyphen rule does not work
OVERRIDES = {
    "Tauros-Paldea-Combat": "tauros-paldea-combat-breed", "Tauros-Paldea-Blaze": "tauros-paldea-blaze-breed",
    "Tauros-Paldea-Aqua": "tauros-paldea-aqua-breed", "Meowstic-F": "meowstic-female", "Meowstic": "meowstic-male",
    "Indeedee-F": "indeedee-female", "Indeedee": "indeedee-male", "Basculegion-F": "basculegion-female",
    "Basculegion": "basculegion-male", "Maushold-Four": "maushold-family-of-four",
    "Maushold": "maushold-family-of-three",
    "Lycanroc": "lycanroc-midday", "Toxtricity": "toxtricity-amped", "Gourgeist": "gourgeist-average",
    "Gourgeist-Small": "gourgeist-small", "Gourgeist-Large": "gourgeist-large", "Gourgeist-Super": "gourgeist-super",
    "Mimikyu": "mimikyu-disguised", "Aegislash": "aegislash-shield", "Palafin": "palafin-zero",
    "Tatsugiri": "tatsugiri-curly", "Squawkabilly": "squawkabilly-green-plumage",
    "Squawkabilly-Blue": "squawkabilly-blue-plumage", "Squawkabilly-Yellow": "squawkabilly-yellow-plumage",
    "Squawkabilly-White": "squawkabilly-white-plumage", "Morpeko": "morpeko-full-belly", "Eiscue": "eiscue-ice",
    "Wishiwashi": "wishiwashi-solo", "Minior": "minior-red-meteor", "Oricorio": "oricorio-baile",
    "Pumpkaboo": "pumpkaboo-average", "Zygarde": "zygarde-50", "Darmanitan": "darmanitan-standard",
}


def api_name(name: str) -> str:
    return OVERRIDES.get(name, name.lower().replace(" ", "-").replace(".", "").replace("'", "").replace(":", ""))


def fetch_json(session: httpx.Client, url: str) -> dict | None:
    time.sleep(DELAY)
    r = session.get(url, timeout=30)
    return r.json() if r.is_success else None


def main() -> None:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    legal = json.loads((SPRITE_DIR / "legal.json").read_text())
    session = httpx.Client(headers={"User-Agent": "poke-sim (personal research; caches locally)"},
                           follow_redirects=True)
    missing = []
    for sp in legal:
        out = HOME_DIR / f"{sp['id']}.png"
        if out.exists():
            continue
        data = fetch_json(session, f"{API}/pokemon/{api_name(sp['name'])}")
        if data is None:  # cosmetic formes (Vivillon, Alcremie, ...) are pokemon-forms: use the species default
            species = fetch_json(session, f"{API}/pokemon-species/{sp['num']}")
            default = next((v["pokemon"]["url"] for v in (species or {}).get("varieties", []) if v["is_default"]), None)
            data = fetch_json(session, default) if default else None
        url = ((data or {}).get("sprites", {}).get("other", {}).get("home", {}) or {}).get("front_default")
        if not url:
            missing.append(sp["name"])
            continue
        time.sleep(DELAY)
        img = session.get(url, timeout=30)
        if img.is_success:
            out.write_bytes(img.content)
        else:
            missing.append(sp["name"])
    print(f"{len(list(HOME_DIR.glob('*.png')))} references in {HOME_DIR}; missing: {missing}")


if __name__ == "__main__":
    main()
