"""Training one fixed team: its roster features, counter-picking opponents and move priors for disruption.

Features added on top of pokesim.rl.features.embed:
- which roster member sits in each team slot (one-hot), so the policy can learn member-specific plans;
- the chance of carrying each kind of move that ruins a Baton Pass / set-up plan (Encore, phazing, Taunt, Haze,
  Yawn) or the Unaware ability, for: the opposing active Pokémon; the worst case over its other seen members;
  and some not-yet-seen brought member (from the 6 shown at team preview). Per Pokémon it is 1 when revealed,
  0 when all four moves are known without it, otherwise the usage-stats prior.

Opponents pick their 3 after seeing our 6; `counter_preview` prefers members carrying those moves.

Usage (build the priors from the Smogon stats in the DB): uv run python -m pokesim.rl.myteam
"""
import json
import random
from pathlib import Path

import numpy as np
from poke_env.battle import AbstractBattle, PokemonType
from poke_env.teambuilder import Teambuilder

from pokesim.config import PROCESSED_DIR

COUNTER_GROUPS = {
    "encore": {"encore"},
    "phaze": {"roar", "whirlwind", "dragontail", "circlethrow"},
    "taunt": {"taunt"},
    "haze": {"haze", "clearsmog"},
    "yawn": {"yawn"},
}
COUNTER_ABILITIES = {"unaware"}
PRIORS_PATH = PROCESSED_DIR / "counter_priors.json"


def to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


class FixedTeam(Teambuilder):
    """Always the same 6 sets (dicts as in data/my_teams/*.json), in the given order."""

    def __init__(self, sets: list[dict]):
        self.sets = sets
        self.packed = self.join_team(self.parse_showdown_team(self.to_showdown(sets)))

    @staticmethod
    def to_showdown(sets: list[dict]) -> str:
        names = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}
        blocks = []
        for s in sets:
            evs = " / ".join(f"{v} {names[k]}" for k, v in s["evs"].items() if v)
            blocks.append(f"{s['species']} @ {s['item']}\nAbility: {s['ability']}\nLevel: 50\nEVs: {evs}\n"
                          f"{s['nature']} Nature\n" + "\n".join(f"- {m}" for m in s["moves"]))
        return "\n\n".join(blocks)

    def yield_team(self) -> str:
        return self.packed


def load_my_team(path: str | Path, lineup: list[str]) -> list[dict]:
    """The team's sets with the chosen lineup first (so '/team 123' brings it, leading with lineup[0])."""
    sets = json.loads(Path(path).read_text())
    order = [s for name in lineup for s in sets if s["species"] == name]
    return order + [s for s in sets if s not in order]


def build_priors() -> dict[str, dict[str, float]]:
    from pokesim.data.smogon_stats import smogon_summaries

    out = {}
    for sid, sm in smogon_summaries().items():
        total = sum(sm.natures.values())
        if not total:
            continue
        probs = {g: min(1.0, sum(sm.moves.get(m, 0) for m in moves) / total) for g, moves in COUNTER_GROUPS.items()}
        ab_total = sum(sm.abilities.values()) or 1
        probs["unaware"] = sum(v for k, v in sm.abilities.items() if to_id(k) in COUNTER_ABILITIES) / ab_total
        out[sid] = {k: round(v, 3) for k, v in probs.items()}
    return out


def load_priors() -> dict[str, dict[str, float]]:
    return json.loads(PRIORS_PATH.read_text()) if PRIORS_PATH.exists() else {}


class TeamFeatures:
    """Extra observation features for a fixed roster (see module docstring)."""

    def __init__(self, roster: list[str], priors: dict[str, dict[str, float]] | None = None):
        self.roster = [to_id(r) for r in roster]
        self.priors = load_priors() if priors is None else priors
        self.n_counter = len(COUNTER_GROUPS) + 1  # + Unaware
        self.size = 6 * len(self.roster) + 3 * self.n_counter

    def counter_probs(self, mon) -> list[float]:
        prior = self.priors.get(to_id(mon.base_species), {}) or self.priors.get(to_id(mon.species), {})
        known = set(mon.moves)
        out = []
        for group, moves in COUNTER_GROUPS.items():
            if known & moves:
                out.append(1.0)
            elif len(known) >= 4:
                out.append(0.0)
            else:
                out.append(prior.get(group, 0.0))
        ability = to_id(mon.ability or "")
        out.append(float(ability in COUNTER_ABILITIES) if ability else prior.get("unaware", 0.0))
        return out

    def roster_index(self, mon) -> int:
        for key in (to_id(mon.species), to_id(mon.base_species)):
            for i, r in enumerate(self.roster):
                if key == r or key.startswith(r):
                    return i
        return -1

    def __call__(self, battle: AbstractBattle) -> np.ndarray:
        feats = np.zeros(self.size, dtype=np.float32)
        for slot, mon in enumerate(list(battle.team.values())[:6]):
            idx = self.roster_index(mon)
            if idx >= 0:
                feats[slot * len(self.roster) + idx] = 1.0
        base = 6 * len(self.roster)
        n = self.n_counter
        opp = battle.opponent_active_pokemon
        if opp is not None:
            feats[base:base + n] = self.counter_probs(opp)
        others = [m for m in battle.opponent_team.values() if not m.fainted and m is not opp]
        if others:
            feats[base + n:base + 2 * n] = np.max([self.counter_probs(m) for m in others], axis=0)
        # unseen members: each previewed one not yet seen was brought with chance (slots left) / (unseen count)
        seen = {to_id(m.base_species) for m in battle.opponent_team.values()}
        unseen = [m for m in battle.teampreview_opponent_team if to_id(m.base_species) not in seen]
        left = max(0, 3 - len(battle.opponent_team))
        if unseen and left:
            p_brought = left / len(unseen)
            probs = np.array([self.counter_probs(m) for m in unseen]) * p_brought
            feats[base + 2 * n:base + 3 * n] = 1.0 - np.prod(1.0 - probs, axis=0)
        return feats


# Types that hit Scolipede (Bug/Poison) super-effectively, and so threaten the passer through Iron Defense.
PASSER_WEAK = {PokemonType.FIRE, PokemonType.FLYING, PokemonType.PSYCHIC, PokemonType.ROCK}


def counter_score(mon) -> float:
    """How strongly an opponent would want to bring `mon` against a Baton Pass team (it knows its own set)."""
    moves = set(mon.moves)
    score = 1.0
    score += 2.0 * any(moves & g for g in COUNTER_GROUPS.values())
    score += 2.0 * (to_id(mon.ability or "") in COUNTER_ABILITIES)
    score += 0.5 * any(m.base_power and m.type in PASSER_WEAK for m in mon.moves.values())
    return score


def counter_preview(battle: AbstractBattle, rng: random.Random, p_counter: float = 0.7) -> str:
    """Team preview order: with probability p_counter, 3 members sampled by counter_score, else 3 at random."""
    mons = list(battle.team.values())
    idx = list(range(len(mons)))
    if rng.random() < p_counter:
        weights = [counter_score(m) for m in mons]
        picks = []
        for _ in range(min(3, len(idx))):
            i = rng.choices(idx, weights=[weights[j] for j in idx])[0]
            picks.append(i)
            idx.remove(i)
    else:
        picks = rng.sample(idx, min(3, len(idx)))
    rest = [i for i in range(len(mons)) if i not in picks]
    return "/team " + "".join(str(i + 1) for i in picks + rest)


class RandomLineup:
    """Picklable team preview for our side: a random 3 of the 6 (random lead) every battle."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, battle: AbstractBattle) -> str:
        picks = self.rng.sample(range(6), 3)
        return "/team " + "".join(str(i + 1) for i in picks + [i for i in range(6) if i not in picks])


class CounterPreview:
    """Picklable `opponent_preview` for the environment (see counter_preview)."""

    def __init__(self, p_counter: float = 0.7, seed: int = 0):
        self.p_counter, self.rng = p_counter, random.Random(seed)

    def __call__(self, battle: AbstractBattle) -> str:
        return counter_preview(battle, self.rng, self.p_counter)


def main() -> None:
    priors = build_priors()
    PRIORS_PATH.write_text(json.dumps(priors, ensure_ascii=False, indent=0))
    top = sorted(priors.items(), key=lambda kv: -max(kv[1].values()))[:10]
    print(f"wrote {len(priors)} species to {PRIORS_PATH}")
    for sid, p in top:
        print(f"  {sid}: {p}")


if __name__ == "__main__":
    main()
