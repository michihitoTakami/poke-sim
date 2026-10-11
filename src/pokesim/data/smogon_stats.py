"""Per-species set priors from Smogon's monthly usage statistics (chaos JSON) for the Showdown Champions format.

Mega formes are folded into their base species (their stone becomes the item); abilities listed under a
mega forme are skipped because the team needs the pre-evolution ability.

Usage: uv run python -m pokesim.data.smogon_stats [--month 2026-09] [--cutoff 1500]
"""
import argparse
import json
import re

import httpx
from sqlalchemy import select

from pokesim.config import FORMAT, RAW_DIR
from pokesim.data.summarize_sets import SpeciesSummary
from pokesim.db.models import Species
from pokesim.db.session import Session

URL = "https://www.smogon.com/stats/{month}/chaos/{format}-{cutoff}.json"
DEFAULT_MONTH, DEFAULT_CUTOFF = "2026-09", 1500


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_chaos(month: str = DEFAULT_MONTH, cutoff: int = DEFAULT_CUTOFF, format_id: str = FORMAT) -> dict:
    path = RAW_DIR / "smogon" / f"{month}-{format_id}-{cutoff}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(URL.format(month=month, format=format_id, cutoff=cutoff), timeout=60)
        path.write_bytes(resp.raise_for_status().content)
    return json.loads(path.read_text())


def smogon_summaries(month: str = DEFAULT_MONTH, cutoff: int = DEFAULT_CUTOFF) -> dict[str, SpeciesSummary]:
    chaos = load_chaos(month, cutoff)["data"]
    with Session() as s:
        species = {sp.id: sp for sp in s.scalars(select(Species))}
    out: dict[str, SpeciesSummary] = {}
    for name, entry in chaos.items():
        sp = species.get(to_id(name))
        if sp is None:
            continue
        is_mega = sp.is_mega and sp.changes_from
        sm = out.setdefault(to_id(sp.changes_from) if is_mega else sp.id,
                            SpeciesSummary(to_id(sp.changes_from) if is_mega else sp.id))
        sm.teams += round(entry["Raw count"])
        sm.items.update({k: v for k, v in entry["Items"].items() if k and k != "nothing"})
        if not is_mega:
            sm.abilities.update({k: v for k, v in entry["Abilities"].items() if k})
        sm.moves.update({k: v for k, v in entry["Moves"].items() if k})
        for spread, weight in entry["Spreads"].items():
            nature, points = spread.split(":")
            points = points.replace("/", "-")
            sm.natures[to_id(nature)] += weight
            sm.spreads[points] += weight
            sm.nature_spreads[f"{to_id(nature)}:{points}"] += weight
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=DEFAULT_MONTH)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    args = parser.parse_args()
    for sm in sorted(smogon_summaries(args.month, args.cutoff).values(), key=lambda x: -x.teams)[:10]:
        print(sm.species_id, sm.teams, sm.items.most_common(2), sm.nature_spreads.most_common(1),
              [m for m, _ in sm.moves.most_common(4)])


if __name__ == "__main__":
    main()
