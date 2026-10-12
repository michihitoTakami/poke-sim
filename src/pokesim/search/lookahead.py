"""P4 lookahead: score every (our choice, their choice) pair on the Showdown sim and solve the resulting matrix game.

The opponent's unknown sets are sampled from the set priors (article sets and Smogon usage stats) consistent with
what the battle revealed, and their unrevealed picks are drawn from the team preview. Each sample gives a payoff
matrix (from services/sim-tools/lookahead.js); we solve it as a zero-sum game, then average over samples:
  - prob[a]:  our equilibrium mix, averaged over samples (the recommended action distribution)
  - value[a]: expected score of action a against the opponent's equilibrium mix (HP balance; a win counts +3)
"""
import json
import random
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np
from poke_env.battle import Battle, Pokemon
from sqlalchemy import select

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.data.smogon_stats import smogon_summaries
from pokesim.data.summarize_sets import SpeciesSummary, summarize
from pokesim.db.models import Item, Learnset, Move, Species
from pokesim.db.session import Session

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
WEATHER_IDS = {"RAINDANCE": "raindance", "SUNNYDAY": "sunnyday", "SANDSTORM": "sandstorm", "SNOWSCAPE": "snowscape",
               "HAIL": "snowscape"}
TERRAIN_IDS = {"ELECTRIC_TERRAIN": "electricterrain", "GRASSY_TERRAIN": "grassyterrain",
               "MISTY_TERRAIN": "mistyterrain", "PSYCHIC_TERRAIN": "psychicterrain"}
SIDE_IDS = {"STEALTH_ROCK": ("stealthrock", None), "SPIKES": ("spikes", None), "TOXIC_SPIKES": ("toxicspikes", None),
            "STICKY_WEB": ("stickyweb", None), "REFLECT": ("reflect", 5), "LIGHT_SCREEN": ("lightscreen", 5),
            "AURORA_VEIL": ("auroraveil", 5), "TAILWIND": ("tailwind", 4)}
VOLATILE_IDS = {"SUBSTITUTE": "substitute", "LEECH_SEED": "leechseed", "YAWN": "yawn", "TAUNT": "taunt",
                "CONFUSION": "confusion"}
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class Advice:
    labels: list[str]  # "move:<id>", "move:<id>:mega", "switch:<species id>"
    prob: np.ndarray
    value: np.ndarray
    opponent: dict[str, float] = field(default_factory=dict)  # opponent's equilibrium mix, averaged over samples

    def ranked(self) -> list[tuple[str, float, float]]:
        order = np.argsort(-self.prob)
        return [(self.labels[i], float(self.prob[i]), float(self.value[i])) for i in order]


def solve_zero_sum(payoff: np.ndarray, iters: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Approximate equilibrium (row maximizes) by regret matching+; returns (row mix, column mix)."""
    n, m = payoff.shape
    r_reg, c_reg = np.zeros(n), np.zeros(m)
    r_sum, c_sum = np.zeros(n), np.zeros(m)
    for t in range(1, iters + 1):
        x = r_reg / r_reg.sum() if r_reg.sum() > 0 else np.full(n, 1 / n)
        y = c_reg / c_reg.sum() if c_reg.sum() > 0 else np.full(m, 1 / m)
        row_u, col_u = payoff @ y, -(x @ payoff)
        r_reg = np.maximum(r_reg + row_u - x @ row_u, 0)
        c_reg = np.maximum(c_reg + col_u - y @ col_u, 0)
        r_sum += t * x
        c_sum += t * y
    return r_sum / r_sum.sum(), c_sum / c_sum.sum()


class SetSampler:
    """Samples plausible full sets for an opponent's Pokémon given what has been revealed."""

    def __init__(self, seasons: list[str], seed: int = 0):
        articles = {x.species_id: x for x in summarize(seasons)}
        smogon = smogon_summaries()
        self.summary: dict[str, SpeciesSummary] = {}
        for sid in set(articles) | set(smogon):
            cands = [t[sid] for t in (articles, smogon) if sid in t]
            self.summary[sid] = next((c for c in cands if c.teams >= 3), cands[0])
        with Session() as s:
            self.species = {sp.id: sp for sp in s.scalars(select(Species))}
            self.moves = {m.id: m for m in s.scalars(select(Move))}
            self.items = {i.id: i for i in s.scalars(select(Item))}
            self.learnsets: dict[str, set[str]] = {}
            for sid, mid in s.execute(select(Learnset.species_id, Learnset.move_id)):
                self.learnsets.setdefault(sid, set()).add(mid)
        self.rng = random.Random(seed)

    def base_id(self, species_id: str) -> str:
        """Team-sheet species for an in-battle forme: megas and battle-only formes (Mimikyu-Busted) -> base."""
        sp = self.species.get(species_id)
        if sp is None:
            return species_id
        if sp.battle_only:
            return to_id(sp.battle_only)
        return to_id(sp.changes_from) if sp.is_mega and sp.changes_from else species_id

    def _weighted(self, counter, exclude=()) -> str | None:
        items = [(k, v) for k, v in counter.items() if k not in exclude and v > 0]
        if not items:
            return None
        keys, weights = zip(*items)
        return self.rng.choices(keys, weights=weights)[0]

    def sample(self, species_id: str, moves=(), item: str | None = None, ability: str | None = None) -> dict:
        base = self.base_id(species_id)
        sp, sm = self.species[base], self.summary.get(base)
        learnable = self.learnsets.get(base, set())
        out = [m for m in moves if m in learnable][:4]
        while sm and len(out) < 4 and (m := self._weighted(sm.moves, exclude=set(out) | {"", "struggle"})):
            if m in learnable:
                out.append(m)
            else:
                sm.moves[m] = 0
        if len(out) < 4:  # fallback: strongest STAB-weighted moves
            def power(mid: str) -> float:
                mv = self.moves[mid]
                return mv.base_power * (1.5 if mv.type in sp.types else 1.0) * (mv.accuracy or 100) / 100
            out += sorted((m for m in learnable if m not in out and self.moves[m].base_power > 0),
                          key=power, reverse=True)[:4 - len(out)]
        if item is None:
            item = self._weighted(sm.items) if sm else None
        if item and self.items.get(item) and self.items[item].mega_stone:
            mega = self.species.get(self.items[item].mega_stone)
            if not mega or mega.changes_from != sp.name:
                item = None
        if ability not in sp.abilities:
            ability = (self._weighted({a: n for a, n in sm.abilities.items() if a in sp.abilities})
                       if sm else None) or sp.abilities[0]
        joint = self._weighted(sm.nature_spreads) if sm and sm.nature_spreads else None
        nature, points = joint.split(":") if joint else ("serious", "2-32-0-0-0-32")
        return {"species": sp.name, "item": self.items[item].name if item in self.items else "",
                "ability": ability, "nature": nature, "moves": out,
                "evs": dict(zip(STATS, map(int, points.split("-")))), "level": 50}


def mon_state(mon: Pokemon, active: bool) -> dict:
    out = {"hp": 0.0 if mon.fainted else mon.current_hp_fraction}
    if mon.status and not mon.fainted:
        out["status"] = mon.status.name.lower()
    if active:
        out["boosts"] = {k: v for k, v in mon.boosts.items() if k in BOOSTS and v}
        out["volatiles"] = [VOLATILE_IDS[e.name] for e in mon.effects if e.name in VOLATILE_IDS]
        out["mega"] = "mega" in mon.species
    return out


def side_conditions(conds: dict, turn: int) -> dict:
    out = {}
    for cond, value in conds.items():
        if cond.name in SIDE_IDS:
            sid, duration = SIDE_IDS[cond.name]
            out[sid] = value if duration is None else max(1, duration - (turn - value))
    return out


def field_state(battle: Battle) -> dict:
    out = {}
    for w, start in battle.weather.items():
        if w.name in WEATHER_IDS:
            out["weather"], out["weatherTurns"] = WEATHER_IDS[w.name], max(1, 5 - (battle.turn - start))
    for f, start in battle.fields.items():
        if f.name in TERRAIN_IDS:
            out["terrain"], out["terrainTurns"] = TERRAIN_IDS[f.name], max(1, 5 - (battle.turn - start))
        elif f.name == "TRICK_ROOM":
            out["trickroom"] = max(1, 5 - (battle.turn - start))
    return out


class Lookahead:
    """Runs the Node lookahead worker and turns battles into recommendations."""

    def __init__(self, sampler: SetSampler, samples: int = 6, seeds: int = 2, rollout: int = 3):
        self.sampler, self.n_samples, self.seeds, self.rollout = sampler, samples, seeds, rollout
        self.proc = subprocess.Popen(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def close(self) -> None:
        self.proc.kill()

    def _query(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        reply = json.loads(self.proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def own_side(self, battle: Battle, own_sets: dict[str, dict]) -> dict:
        active = battle.active_pokemon
        bench = [m for m in battle.team.values() if m is not active and (m.fainted or m in battle.available_switches)]
        if not battle.available_switches and not battle.trapped:
            bench = [m for m in battle.team.values() if m is not active][:2]
        mons = [active] + bench[:2]
        return {"team": [own_sets[self.sampler.base_id(m.species)] for m in mons],
                "mons": [mon_state(m, i == 0) for i, m in enumerate(mons)],
                "megaUsed": battle.used_mega_evolve, "conditions": side_conditions(battle.side_conditions,
                                                                                    battle.turn)}

    def opponent_side(self, battle: Battle) -> dict:
        active = battle.opponent_active_pokemon
        seen = [active] + [m for m in battle.opponent_team.values() if m is not active]
        seen_ids = {self.sampler.base_id(m.species) for m in seen}
        unseen = [m.species for m in battle.teampreview_opponent_team
                  if self.sampler.base_id(m.species) not in seen_ids]
        self.sampler.rng.shuffle(unseen)
        team, mons = [], []
        for i, m in enumerate(seen[:3]):
            item = None if m.item == "unknown_item" else (m.item or "")
            if "mega" in m.species and not item:
                item = None
            team.append(self.sampler.sample(m.species, list(m.moves), item, m.ability))
            mons.append(mon_state(m, i == 0))
        for sid in unseen[:3 - len(team)]:
            team.append(self.sampler.sample(sid))
            mons.append({"hp": 1.0})
        return {"team": team, "mons": mons, "megaUsed": battle.opponent_used_mega_evolve,
                "conditions": side_conditions(battle.opponent_side_conditions, battle.turn)}

    def advise(self, battle: Battle, own_sets: dict[str, dict]) -> Advice:
        own = self.own_side(battle, own_sets)
        field = field_state(battle)
        samples = [{"p1": own, "p2": self.opponent_side(battle), "field": field} for _ in range(self.n_samples)]
        results = self._query({"format": FORMAT, "seeds": self.seeds, "rollout": self.rollout,
                               "samples": samples})["results"]
        labels = results[0]["p1"]
        prob, value = np.zeros(len(labels)), np.zeros(len(labels))
        opponent: dict[str, float] = {}
        for res in results:
            payoff = np.array(res["payoff"])
            idx = [res["p1"].index(lab) for lab in labels]  # our options are the same in every sample
            x, y = solve_zero_sum(payoff[idx])
            prob += x / len(results)
            value += payoff[idx] @ y / len(results)
            for lab, q in zip(res["p2"], y):  # the opponent's options depend on its sampled set
                opponent[lab] = opponent.get(lab, 0.0) + float(q) / len(results)
        return Advice(labels, prob, value, opponent)


def label_to_action(label: str, battle: Battle) -> int | None:
    """Map a lookahead label to poke-env's SinglesEnv action index (0-5 switch, 6-9 move, 10-13 move+mega)."""
    kind, _, rest = label.partition(":")
    if kind == "switch":
        for i, mon in enumerate(battle.team.values()):
            if to_id(mon.species) == rest or mon.species == rest:
                return i
        return None
    move_id, _, mega = rest.partition(":")
    moves = list(battle.active_pokemon.moves.keys())[:4]
    if move_id not in moves:
        return None
    return (10 if mega else 6) + moves.index(move_id)
