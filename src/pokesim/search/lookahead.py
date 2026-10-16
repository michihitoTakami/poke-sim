"""P4 lookahead: score every (our choice, their choice) pair on the Showdown sim and solve the resulting matrix game.

The opponent's unknown sets are sampled from the set priors (article sets and Smogon usage stats) consistent with
what the battle revealed, and their unrevealed picks are drawn from the team preview. Each sample gives a payoff
matrix (from services/sim-tools/lookahead.js); we solve it as a zero-sum game, then average over samples:
  - prob[a]:  our equilibrium mix, averaged over samples (the recommended action distribution)
  - value[a]: expected score of action a against the opponent's equilibrium mix (HP balance; a win counts +3)
"""
import hashlib
import json
import os
import random
import re
import select as io_select
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
from poke_env.battle import Battle, Pokemon
from poke_env.data import GenData
from sqlalchemy import select

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.data.smogon_stats import smogon_summaries
from pokesim.data.summarize_sets import SpeciesSummary, summarize
from pokesim.db.models import Item, Learnset, Move, Species
from pokesim.db.session import Session
from pokesim.sim.formes import current_forme

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
WEATHER_IDS = {"RAINDANCE": "raindance", "SUNNYDAY": "sunnyday", "SANDSTORM": "sandstorm", "SNOWSCAPE": "snowscape",
               "HAIL": "snowscape"}
TERRAIN_IDS = {"ELECTRIC_TERRAIN": "electricterrain", "GRASSY_TERRAIN": "grassyterrain",
               "MISTY_TERRAIN": "mistyterrain", "PSYCHIC_TERRAIN": "psychicterrain"}
SIDE_IDS = {"STEALTH_ROCK": ("stealthrock", None), "SPIKES": ("spikes", None), "TOXIC_SPIKES": ("toxicspikes", None),
            "STICKY_WEB": ("stickyweb", None), "REFLECT": ("reflect", 5), "LIGHT_SCREEN": ("lightscreen", 5),
            "AURORA_VEIL": ("auroraveil", 5), "TAILWIND": ("tailwind", 4)}
VOLATILE_IDS = {"SUBSTITUTE": "substitute", "LEECH_SEED": "leechseed", "YAWN": "yawn", "TAUNT": "taunt",
                "CONFUSION": "confusion", "CURSE": "curse", "SALT_CURE": "saltcure", "AQUA_RING": "aquaring",
                "INGRAIN": "ingrain", "FOCUS_ENERGY": "focusenergy", "MAGNET_RISE": "magnetrise",
                "NO_RETREAT": "noretreat", "HEAL_BLOCK": "healblock", "TORMENT": "torment"}
PERISH = {"PERISH0": 0, "PERISH1": 1, "PERISH2": 2, "PERISH3": 3}
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def mega_evolved(mon) -> bool:
    """poke-env keeps the species id on a Mega forme change (|detailschange|) and only swaps stats, types and ability:
    a Mega shows as stats that differ from its own species entry."""
    if "mega" in mon.species:
        return True
    from poke_env.data import GenData

    entry = GenData.from_gen(9).pokedex.get(mon.species)
    return bool(entry) and bool(mon.base_stats) and dict(entry["baseStats"]) != dict(mon.base_stats)


@dataclass
class Advice:
    labels: list[str]  # "move:<id>", "move:<id>:mega", "switch:<species id>"
    prob: np.ndarray
    value: np.ndarray
    opponent: dict[str, float] = field(default_factory=dict)  # opponent's equilibrium mix, averaged over samples
    status: dict = field(default_factory=dict)  # how it was computed: failed branches, solver, ...
    worst: np.ndarray | None = None  # per action, the weighted worst case over the opponent's options (planner)

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


OBSERVED_WEIGHT = 5.0   # one battle of ours = this many published teams
UTILITY_PSEUDO = 0.03   # share of teams given to each learnable utility move
UTILITY_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "yawn", "curse", "substitute", "protect",
                 "taunt", "encore", "willowisp", "toxic", "thunderwave", "glare", "roar", "whirlwind", "dragontail",
                 "haze", "clearsmog", "trick", "uturn", "voltswitch", "flipturn", "partingshot", "knockoff",
                 "rapidspin", "defog", "recover", "roost", "slackoff", "synthesis", "moonlight", "painsplit",
                 "leechseed", "swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup", "irondefense",
                 "trickroom", "tailwind", "destinybond", "perishsong", "disable", "suckerpunch", "fakeout",
                 "nuzzle", "healbell", "reflect", "lightscreen", "batonpass", "rest", "sleeptalk"}


def load_observed() -> dict:
    try:
        from pokesim.live.observed import load

        return load()
    except Exception:  # noqa: BLE001 - no records yet
        return {}


class SetSampler:
    """Samples plausible full sets for an opponent's Pokémon given what has been revealed."""

    def __init__(self, seasons: list[str], seed: int = 0, observed: dict | None = None,
                 observed_weight: float = OBSERVED_WEIGHT, utility: float = UTILITY_PSEUDO,
                 summary: dict[str, SpeciesSummary] | None = None):
        self.observed_weight, self.utility = observed_weight, utility
        self.summary: dict[str, SpeciesSummary] = {} if summary is None else summary
        if summary is None:
            articles = {x.species_id: x for x in summarize(seasons)}
            smogon = smogon_summaries()
            for sid in set(articles) | set(smogon):
                cands = [t[sid] for t in (articles, smogon) if sid in t]
                self.summary[sid] = next((c for c in cands if c.teams >= 3), cands[0])
        self.observed = self._blend_observed(observed if observed is not None else load_observed())
        with Session() as s:
            self.species = {sp.id: sp for sp in s.scalars(select(Species))}
            self.moves = {m.id: m for m in s.scalars(select(Move))}
            self.items = {i.id: i for i in s.scalars(select(Item))}
            self.learnsets: dict[str, set[str]] = {}
            for sid, mid in s.execute(select(Learnset.species_id, Learnset.move_id)):
                self.learnsets.setdefault(sid, set()).add(mid)
        self.rng = random.Random(seed)

    def _blend_observed(self, observed: dict) -> dict:
        """Sets seen in our own battles count as `observed_weight` teams each (they are the current ladder); species
        only we have seen get a summary of their own."""
        for sid, obs in observed.items():
            if not obs.get("brought"):
                continue
            sm = self.summary.setdefault(sid, SpeciesSummary(species_id=sid))
            w = self.observed_weight
            sm.teams += round(w * obs["brought"])
            for key in ("moves", "items", "abilities"):
                for k, n in obs[key].items():
                    getattr(sm, key)[k] += w * n
        return observed

    def _with_utility(self, base: str, sm: SpeciesSummary | None) -> dict[str, float]:
        """Move weights: the summary plus a small share for common utility moves it can learn, so unlisted
        Yawn / Curse / Substitute / Stealth Rock are possible instead of impossible."""
        moves = dict(sm.moves) if sm else {}
        teams = max(sm.teams if sm else 0, 1)
        for m in UTILITY_MOVES & self.learnsets.get(base, set()):
            moves[m] = moves.get(m, 0) + self.utility * teams
        return moves

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
        weights = self._with_utility(base, sm)
        while len(out) < 4 and (m := self._weighted(weights, exclude=set(out) | {"", "struggle"})):
            if m in learnable:
                out.append(m)
            else:
                weights[m] = 0
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


def revealed(mon: Pokemon) -> tuple[tuple[str, ...], str | None, str | None]:
    """An opponent's Pokémon's revealed moves, item ("" when known to have none, None when unknown) and ability."""
    item = None if mon.item == "unknown_item" else (mon.item or "")
    if mega_evolved(mon) and not item:
        item = None
    return tuple(mon.moves), item, mon.ability


def mon_state(mon: Pokemon, active: bool, own: bool = False) -> dict:
    from pokesim.state.codec import known_counters

    out = {"hp": 0.0 if mon.fainted else mon.current_hp_fraction, "mega": mega_evolved(mon)}
    out.update(known_counters(mon, active, own))
    entry = GenData.from_gen(9).pokedex.get(current_forme(mon), {})
    if entry.get("battleOnly") and not mega_evolved(mon):  # e.g. Mimikyu-Busted (no Disguise left), Aegislash-Blade
        out["forme"] = entry["name"]
    if mon.status and not mon.fainted:
        out["status"] = mon.status.name.lower()
    if active:
        out["boosts"] = {k: v for k, v in mon.boosts.items() if k in BOOSTS and v}
        out["volatiles"] = [VOLATILE_IDS[e.name] for e in mon.effects if e.name in VOLATILE_IDS]
        perish = [PERISH[e.name] for e in mon.effects if e.name in PERISH]
        if perish:  # Perish Song: faints when the count runs out
            out["perish"] = min(perish)
        if any(e.name == "ENCORE" for e in mon.effects) and mon.last_move is not None:
            out["encore"] = mon.last_move.id  # locked into its last move
        if mon.protect_counter:  # consecutive Protect / Detect / Endure: the next one succeeds 1 in 3^n
            out["stall"] = mon.protect_counter
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

    def __init__(self, sampler: SetSampler, samples: int = 6, seeds: int = 2, rollout: int = 3,
                 opponent_model: dict | None = None, win_decay: float | None = None, search: dict | None = None,
                 value_model=None):
        """search: worker settings for every query (see services/sim-tools/lookahead.js): continuations, scoring,
        lostPathPenalty, deadlineMs, and seeds / rollout / winDecay overriding the arguments."""
        self.sampler, self.n_samples, self.seeds, self.rollout = sampler, samples, seeds, rollout
        self.win_decay = win_decay  # None: the worker's default (a quicker win scores higher)
        self.search = dict(search or {})
        self.value_model = value_model
        self.opponent_model = opponent_model or {}
        self.proc = subprocess.Popen(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait(timeout=5)

    def _query(self, request: dict) -> dict:
        if self.proc.poll() is not None:
            self.proc = subprocess.Popen(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + getattr(self, 'query_timeout', 30.)
        chunks = []
        while True:
            remaining = deadline - time.monotonic()
            ready, _, _ = io_select.select([self.proc.stdout], [], [], max(0, remaining))
            if not ready or remaining <= 0:
                self.proc.kill()
                self.proc.wait(timeout=5)
                raise TimeoutError('search worker exceeded response deadline')
            part = os.read(self.proc.stdout.fileno(), 65536)
            if not part:
                raise RuntimeError('search worker exited before replying')
            chunks.append(part)
            if b'\n' in part:
                break
        reply = json.loads(b''.join(chunks))
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def own_side(self, battle: Battle, own_sets: dict[str, dict], incoming: Pokemon | None = None) -> dict:
        """Keep the outgoing active first; the worker executes incoming's actual switch-in transition."""
        active = battle.active_pokemon
        if active is None:
            raise ValueError("switch reconstruction needs the outgoing active")
        # poke-env retains all six preview members in battle.team. Only the current request identifies the
        # selected team, including fainted or trapped members; available_switches cannot identify that roster.
        request = battle.last_request
        roster = request.get("side", {}).get("pokemon", [])
        if request.get("teamPreview") or not 1 <= len(roster) <= 3:
            raise ValueError("selected team request unavailable")
        try:
            brought = [battle.team[entry["ident"]] for entry in roster]
        except KeyError as e:
            raise ValueError("selected team request has an unknown member") from e
        if len({id(mon) for mon in brought}) != len(brought) or active not in brought:
            raise ValueError("selected team request is inconsistent")
        if incoming is not None and incoming not in brought:
            raise ValueError("incoming Pokemon is not on the selected team")
        mons = [active] + [mon for mon in brought if mon is not active]
        states = [mon_state(m, i == 0, own=True) for i, m in enumerate(mons)]
        request = battle.last_request.get("active") or []
        if request:
            states[0]["pp"] = {m["id"]: m["pp"] for m in request[0].get("moves", []) if "pp" in m}
        if hasattr(battle, "state_tracker"):
            states[0].update(battle.state_tracker.locks(battle.player_role, active.name))
        out = {"team": [own_sets[self.sampler.base_id(m.species)] for m in mons],
                "mons": states,
                "megaUsed": battle.used_mega_evolve, "conditions": side_conditions(battle.side_conditions,
                                                                                    battle.turn)}
        if incoming is not None:
            out["incoming"] = mons.index(incoming)
            if not active.fainted and active.last_move is not None and active.last_move.id == "batonpass":
                out["switchMove"] = "batonpass"
        return out

    def opponent_side(self, battle: Battle, belief=None) -> dict:
        """belief (pokesim.belief.sets.SetBelief): draw the revealed Pokémon's sets from what the battle showed
        instead of from the prior alone."""
        active = battle.opponent_active_pokemon
        seen = [active] + [m for m in battle.opponent_team.values() if m is not active]
        seen_ids = {self.sampler.base_id(m.species) for m in seen}
        unseen = [m.species for m in battle.teampreview_opponent_team
                  if self.sampler.base_id(m.species) not in seen_ids]
        self.sampler.rng.shuffle(unseen)
        team, mons = [], []
        for i, m in enumerate(seen[:3]):
            moves, item, ability = revealed(m)
            if belief is not None:
                team.append(belief.draw(m.species, moves, item, ability))
            else:
                team.append(self.sampler.sample(m.species, list(moves), item, ability))
            mons.append(mon_state(m, i == 0))
            if i == 0 and hasattr(battle, "state_tracker"):
                mons[-1].update(battle.state_tracker.locks(battle.opponent_role, m.name))
        for sid in unseen[:3 - len(team)]:
            team.append(self.sampler.sample(sid))
            mons.append({"hp": 1.0})
        return {"team": team, "mons": mons, "megaUsed": battle.opponent_used_mega_evolve,
                "conditions": side_conditions(battle.opponent_side_conditions, battle.turn)}

    def query(self, battle: Battle, own_sets: dict[str, dict], incoming: Pokemon | None = None,
              search: dict | None = None, belief=None, log=None, our_role='p1', value=False,
              capture: bool = False) -> list[dict]:
        """Payoff matrices (our options x the opponent's) for sampled opponent sets. search: settings for this query
        only (as the constructor's). belief: draw the opponent's sets from it (see opponent_side).
        capture: with the terminal search, return the worker request (worlds, observed history, settings) instead
        of searching, so the exact live position can be replayed later (pokesim.evaluation.positions)."""
        started = time.monotonic()
        own = self.own_side(battle, own_sets, incoming)
        field = field_state(battle)
        settings = dict(self.search, **(search or {}))
        deep = value and settings.get('terminalSearch')
        previous_rng = self.sampler.rng if deep else None
        previous_belief_rng = getattr(belief, 'rng', None)
        if deep:
            # Identical observations draw identical particles; a repeated query must not invent a new opponent.
            seed = int(hashlib.sha256(json.dumps([log, own, field], sort_keys=True).encode()).hexdigest()[:16], 16)
            self.sampler.rng = random.Random(seed)
            if previous_belief_rng is not None:
                belief.rng = random.Random(seed)
        try:
            samples = [{"p1": own, "p2": self.opponent_side(battle, belief), "field": field}
                       for _ in range(self.n_samples)]
        finally:
            if deep:
                self.sampler.rng = previous_rng
                if previous_belief_rng is not None:
                    belief.rng = previous_belief_rng
        request = {"format": FORMAT, "seeds": self.seeds, "rollout": self.rollout, "samples": samples}
        if self.win_decay is not None:
            request["winDecay"] = self.win_decay
        request.update(self.search)
        request.update(search or {})
        request.pop("plans", None)  # plan search settings (see plans), not the payoff query's
        if value:
            if self.value_model is None or not log:
                raise ValueError('learned search requires a value model and observed history')
            if our_role not in ('p1', 'p2'):
                raise ValueError('invalid observer role')
            history = list(log)
            if our_role == 'p2':
                history = [re.sub(r'\bp([12])(?=[a-z]?[:|]|$)',
                                  lambda m: 'p' + ('2' if m[1] == '1' else '1'), line) for line in history]
            request.update(valueLeaves=True, observationLog=history)
            if deep:
                from pokesim.search.terminal import search as terminal_search

                if capture:
                    return request
                return terminal_search(self, request, started=started)
            return self.value_model.apply(self._query(request))
        return self._query(request)["results"]

    def plans(self, battle: Battle, own_sets: dict[str, dict], plans: dict, belief=None) -> dict:
        """Plan search (services/sim-tools/lookahead.js evaluatePlans): each of our candidate plans played to the end
        of the battle in sampled worlds, against the opponent policies in `plans`. Returns the worker's reply."""
        own = self.own_side(battle, own_sets)
        field = field_state(battle)
        samples = [{"p1": own, "p2": self.opponent_side(battle, belief), "field": field}
                   for _ in range(self.n_samples)]
        return self._query({"format": FORMAT, "samples": samples, "plans": plans})

    def advise(self, battle: Battle, own_sets: dict[str, dict], last_move: str | None = None) -> Advice:
        return aggregate(self.query(battle, own_sets), self.opponent_model, last_move)

    def advise_switch(self, battle: Battle, own_sets: dict[str, dict], last_move: str | None = None,
                      solve=None) -> Advice:
        """A switch we must make (after a faint, U-turn, Baton Pass, ...): each Pokémon we can send in is scored by
        the position it starts, the value of our mix there (solve(battle, own_sets, last_move, incoming) -> Advice;
        default: the legacy aggregate). The opponent does not choose at the same time, so the best one gets all the
        weight. After our own U-turn the opponent may still have its move this turn; the position is the start of
        the next turn (an approximation)."""
        labels, values, status = [], [], {"switch": {}}
        if len(battle.available_switches) == 1:  # nothing to compare
            mon = battle.available_switches[0]
            return Advice([f"switch:{to_id(mon.species)}"], np.ones(1), np.zeros(1), {}, status={"only_option": True})
        for mon in battle.available_switches:
            if solve is None:
                adv = aggregate(self.query(battle, own_sets, mon), self.opponent_model, last_move)
            else:
                adv = solve(battle, own_sets, last_move, mon)
            labels.append(f"switch:{to_id(mon.species)}")
            values.append(float(adv.status.get("objective", adv.prob @ adv.value)))
            status["switch"][labels[-1]] = adv.status
        if not labels:
            raise ValueError("no switch available")
        prob = np.zeros(len(labels))
        prob[int(np.argmax(values))] = 1.0
        return Advice(labels, prob, np.array(values), {}, status=status)


def clean(result: dict) -> tuple[np.ndarray, list[str], int]:
    """A lookahead result's payoff with failed branches handled: an opponent option that failed in every pairing is
    dropped (it is not possible in that world); any other failed cell takes the worst value of its column (we do
    not count a failure in our favour). Returns (payoff, opponent labels, number of failed cells)."""
    payoff = np.array([[np.nan if v is None else v for v in row] for row in result["payoff"]], dtype=float)
    labels = list(result["p2"])
    failed = int(np.isnan(payoff).sum())
    keep = ~np.isnan(payoff).all(axis=0)
    if not keep.any():
        raise ValueError("every branch failed")
    payoff, labels = payoff[:, keep], [lab for lab, k in zip(labels, keep) if k]
    col_min = np.nanmin(payoff, axis=0)
    payoff = np.where(np.isnan(payoff), col_min[None, :], payoff)
    return payoff, labels, failed


def opponent_mix(payoff: np.ndarray, x: np.ndarray, y: np.ndarray, model: dict) -> np.ndarray:
    """The opponent's predicted action distribution in one sample.
    eq      the equilibrium mix y (assumes a perfect opponent; tends to over-predict switches)
    br      softmax best response to our equilibrium mix x: players pick what looks best against what we do
    The prediction mixes them: (1 - w) * eq + w * br."""
    w, tau = model.get("br_weight", 0.0), model.get("tau", 0.1)
    if w <= 0:
        return y
    u = -(x @ payoff)  # the opponent's payoff of each option against our mix
    br = np.exp((u - u.max()) / tau)
    return (1 - w) * y + w * br / br.sum()


def aggregate(results: list[dict], opponent_model: dict | None = None, last_move: str | None = None) -> Advice:
    """last_move: the opponent active's previous move ("move:<id>"); players repeat it about half the time, so the
    prediction gives it a share `repeat` (from scripts/eval_records.py on our records)."""
    labels = results[0]["p1"]
    prob, value = np.zeros(len(labels)), np.zeros(len(labels))
    opponent: dict[str, float] = {}
    worlds = []
    failed_cells = 0
    for res in results:
        try:
            payoff, opp, failed = clean(res)
        except ValueError:  # every branch of this sample failed: leave it out
            continue
        failed_cells += failed
        worlds.append((payoff[[res["p1"].index(lab) for lab in labels]], opp))  # our options are common
    if not worlds:
        raise ValueError("every sampled world failed")
    for payoff, opp in worlds:
        x, y = solve_zero_sum(payoff)
        prob += x / len(worlds)
        value += payoff @ y / len(worlds)
        q = opponent_mix(payoff, x, y, opponent_model or {})
        for lab, p in zip(opp, q):  # the opponent's options depend on its sampled set
            opponent[lab] = opponent.get(lab, 0.0) + float(p) / len(worlds)
    repeat = (opponent_model or {}).get("repeat", 0.0)
    if repeat > 0 and last_move and any(k.split(":")[:2] == last_move.split(":")[:2] for k in opponent):
        opponent = {k: (1 - repeat) * p for k, p in opponent.items()}
        opponent[last_move] = opponent.get(last_move, 0.0) + repeat
    return Advice(labels, prob, value, opponent,
                  status={"worlds": len(worlds), "failed_worlds": len(results) - len(worlds),
                          "failed_cells": failed_cells})


def label_to_action(label: str, battle: Battle) -> int | None:
    """Map a lookahead label to poke-env's SinglesEnv action index (0-5 switch, 6-9 move, 10-13 move+mega)."""
    kind, _, rest = label.partition(":")
    if kind == "switch":
        matches = [i for i, mon in enumerate(battle.team.values()) if switch_species_matches(rest, mon)]
        return matches[0] if len(matches) == 1 else None
    move_id, _, mega = rest.partition(":")
    moves = list(battle.active_pokemon.moves.keys())[:4]
    if move_id not in moves:
        return None
    return (10 if mega else 6) + moves.index(move_id)


def switch_species_matches(species: str, mon: Pokemon) -> bool:
    """Worker labels use the current forme; poke-env can keep its original species after changing forme.

    Do not collapse every forme to baseSpecies (Floette-Eternal is distinct from ordinary Floette). A Mega
    alias needs the correct battle-only origin AND its observed stats; an unevolved preview member cannot match.
    """
    species = to_id(species)
    if species in {to_id(mon.species), current_forme(mon)}:
        return True
    entry = GenData.from_gen(mon.gen).pokedex.get(species, {})
    if not (entry.get("forme") or "").startswith("Mega"):
        return False
    origins = entry.get("battleOnly") or entry.get("changesFrom") or entry.get("baseSpecies") or []
    if isinstance(origins, str):
        origins = [origins]
    return (to_id(mon.species) in {to_id(s) for s in origins}
            and dict(mon.base_stats) == entry.get("baseStats"))
