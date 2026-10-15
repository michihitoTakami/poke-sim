"""The opponent's sets as a probability over candidates, updated by what the battle showed.

Candidates for an opponent Pokémon are drawn from the set sampler (pokesim.search.lookahead.SetSampler: article
sets, usage stats and our past battles, conditioned on the moves, item and ability revealed so far), so the prior is
the one the lookahead already used. Each observation (pokesim.belief.observe) is scored per candidate on the
simulator (services/sim-tools/belief.js):

  damage, opponent attacking   our HP is exact: the share of the 16 damage rolls that give the HP lost (a KO: rolls
                               at least the HP it had)
  damage, we attack it         its HP is a percentage: the share of rolls within rounding of the percent lost
  speed                        1 if the candidate's speed explains the order, 1/2 on a tie, 0 otherwise
  no choice                    0 for a Choice item, else 1 (no simulator needed)

and floored, L = (1 - EPS) * P + EPS: effects the reconstruction does not carry (an unrevealed ability of ours,
mid-turn changes, a wrong sample of the rest of its set) must not rule a set out for good. The posterior weight of a
candidate is its prior draw times the product of its likelihoods; the lookahead then draws the opponent's sets from
it (Lookahead.query(..., belief=)).
"""
import hashlib
import json
import random
import subprocess

import numpy as np

from pokesim.belief.observe import Observation, Observer
from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR

EPS = 0.02
CANDIDATES = 48
PERCENT_TOLERANCE = 1.0  # two displayed percentages, each rounded


CHOICE = {"Choice Band", "Choice Specs", "Choice Scarf"}


def likelihood(obs: Observation, result: dict | None, candidate: dict | None = None) -> float:
    """P(observation | candidate) before the floor; None result (position not buildable) is uninformative."""
    if obs.kind == "nochoice":
        return 0.0 if candidate and candidate.get("item") in CHOICE else 1.0
    if result is None:
        return 1.0
    if obs.kind == "speed":
        ours, theirs = result["speeds"]
        if ours == theirs:
            return 0.5
        return float((theirs > ours) == obs.opponent_first)
    rolls = result.get("rolls")
    if not rolls:
        return 0.0  # the candidate could not have damaged (immune, or the move fails for it)
    rolls = np.array(rolls, dtype=float)
    if obs.attacker == "p2":  # it hit us: exact HP
        lost = obs.before - obs.after
        if obs.after <= 0:
            return float((rolls >= obs.before).mean())
        return float((rolls == lost).mean())
    pct = rolls / result["maxhp"] * 100  # we hit it: percentages
    lost = obs.before - obs.after
    if obs.after <= 0:
        return float((pct >= obs.before - PERCENT_TOLERANCE).mean())
    return float((np.abs(pct - lost) <= PERCENT_TOLERANCE).mean())


def key(candidate: dict) -> str:
    return hashlib.sha1(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:16]


class Worker:
    def __init__(self):
        self.proc = subprocess.Popen(["node", str(ROOT / "services/sim-tools/belief.js"), str(SHOWDOWN_DIR)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def run(self, jobs: list[dict]) -> list[list[dict | None]]:
        self.proc.stdin.write(json.dumps({"format": FORMAT, "jobs": jobs}) + "\n")
        reply = json.loads(self.proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["results"]

    def close(self) -> None:
        self.proc.kill()


class SetBelief:
    """One battle's belief over the opponent's sets. feed(log) with the log so far, then draw(...) per sample."""

    def __init__(self, sets: list[dict], sampler, worker: Worker, role: str = "p1", n: int = CANDIDATES,
                 eps: float = EPS, seed: int = 0):
        self.sampler, self.worker, self.n, self.eps, self.seed = sampler, worker, n, eps, seed
        self.observer = Observer(sets, sampler.base_id, role)
        self.cache: dict[tuple[str, str], float] = {}  # (candidate key, observation id) -> likelihood
        self.rng = random.Random(seed)

    @property
    def observations(self) -> list[Observation]:
        return self.observer.observations

    def feed(self, log: list[str]) -> list[Observation]:
        return self.observer.feed(log)

    def candidates(self, species: str, moves=(), item: str | None = None, ability: str | None = None) -> list[dict]:
        """Prior draws given what is revealed; the same draws while nothing new is revealed (cached likelihoods)."""
        base = self.sampler.base_id(species)
        tag = json.dumps([base, sorted(moves), item, ability, self.seed])
        saved = self.sampler.rng
        self.sampler.rng = random.Random(int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16))
        try:
            return [self.sampler.sample(base, list(moves), item, ability) for _ in range(self.n)]
        finally:
            self.sampler.rng = saved

    def weights(self, species: str, cands: list[dict], before: int | None = None) -> np.ndarray:
        """Posterior weights of the candidates (prior draws: equal) from the observations of this species
        (the first `before` observations of the battle, default all)."""
        base = self.sampler.base_id(species)
        obs = [o for o in self.observations[:before] if o.species == base]
        if not obs:
            return np.full(len(cands), 1 / len(cands))
        keys = [key(c) for c in cands]
        unique = {k: c for k, c in zip(keys, cands)}
        for o in obs:
            if o.kind == "nochoice":
                for k, c in unique.items():
                    self.cache[(k, o.id)] = likelihood(o, None, c)
        todo = [o for o in obs if any((k, o.id) not in self.cache for k in unique)]
        if todo:
            ks = list(unique)
            results = self.worker.run([dict(o.job(), candidates=[unique[k] for k in ks]) for o in todo])
            for o, res in zip(todo, results):
                for k, r in zip(ks, res):
                    self.cache[(k, o.id)] = likelihood(o, r)
        loglik = np.array([sum(np.log((1 - self.eps) * self.cache[(k, o.id)] + self.eps) for o in obs)
                           for k in keys])
        w = np.exp(loglik - loglik.max())
        return w / w.sum()

    def predictive(self, obs: Observation, cands: list[dict], weights: np.ndarray) -> float:
        """P(obs) under a weighting of candidates (floored as in the posterior)."""
        res = self.results(obs, cands)
        return float(sum(w * ((1 - self.eps) * likelihood(obs, r, c) + self.eps)
                         for w, r, c in zip(weights, res, cands)))

    def results(self, obs: Observation, cands: list[dict]) -> list[dict | None]:
        if obs.kind == "nochoice":
            return [None] * len(cands)
        return self.worker.run([dict(obs.job(), candidates=cands)])[0]

    def draw(self, species: str, moves=(), item: str | None = None, ability: str | None = None) -> dict:
        cands = self.candidates(species, moves, item, ability)
        w = self.weights(species, cands)
        return cands[self.rng.choices(range(len(cands)), weights=w)[0]]

    def summary(self, species: str, moves=(), item: str | None = None, ability: str | None = None,
                top: int = 3) -> list[tuple[str, float]]:
        """The most likely item / nature / spread combinations, for display."""
        cands = self.candidates(species, moves, item, ability)
        w = self.weights(species, cands)
        agg: dict[str, float] = {}
        for c, p in zip(cands, w):
            label = f"{c['item'] or '-'} / {c['nature']} / {'-'.join(str(c['evs'][s]) for s in c['evs'])}"
            agg[label] = agg.get(label, 0.0) + float(p)
        return sorted(agg.items(), key=lambda kv: -kv[1])[:top]


class Beliefs:
    """Per battle beliefs for a player that plays several battles (keyed by battle tag)."""

    def __init__(self, sets: list[dict], sampler, keep: int = 8):
        self.sets, self.sampler, self.keep = sets, sampler, keep
        self.worker = Worker()
        self.by_battle: dict[str, SetBelief] = {}

    def get(self, tag: str, log: list[str], role: str) -> SetBelief:
        b = self.by_battle.get(tag)
        if b is None:
            b = self.by_battle[tag] = SetBelief(self.sets, self.sampler, self.worker, role)
            while len(self.by_battle) > self.keep:
                self.by_battle.pop(next(iter(self.by_battle)))
        b.feed(log)
        return b

    def close(self) -> None:
        self.worker.close()
