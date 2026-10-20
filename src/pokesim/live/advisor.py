"""Per-decision advice for a live battle: what to do (with probabilities), estimated win chance, and what the
opponent is likely to do next.

Sources, combined when both are available:
- the fixed-team policy (pokesim.rl.train --my-team): action probabilities and its value head (as a rough win chance);
- the lookahead (pokesim.search.lookahead): equilibrium mix over our options and the opponent's options, averaged
  over sets sampled for the opponent from usage data and everything revealed so far.
The opponent's next action mixes the search's prediction with a policy cloned from human players
(pokesim.live.opponent), which needs the battle's protocol lines (`log`).

Planner (feature flag, `planner={"mode": "shadow" | "on", "rho": ...}`): the lookahead's worlds are solved for one
common mix (pokesim.search.planner) with the opponent prediction, cloned policy included, fixed before solving.
"shadow" computes it next to the legacy advice (Suggestion.shadow) without acting on it; "on" acts on it, the
policy only as the fallback. A forced switch is searched too (each Pokémon we can send in, by the position it
starts), and a failed search falls back to the policy (Suggestion.status["fallback"]).
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv
from sb3_contrib import MaskablePPO

from pokesim.rl.advise import action_probs
from pokesim.rl.env import REWARDS
from pokesim.rl.features import embed
from pokesim.rl.myteam import TeamFeatures, to_id
from pokesim.search.lookahead import Lookahead, SetSampler, action_moves, aggregate, label_to_action, revealed
from pokesim.team.roles import win_path

POLICY_WEIGHT = 0.5  # share of the policy in the combined recommendation
# How the policy and the lookahead are combined (see combine()). "linear" was the plain average of the two
# distributions; a near-deterministic policy then decided almost every turn on its own.
COMBINE = {"mode": "product", "weight": POLICY_WEIGHT, "temperature": 2.0, "tau": 0.5, "margin": 1.0, "defer": 0.9}


@dataclass
class Suggestion:
    actions: list[tuple[str, float]]  # (label, probability), best first; labels as in the lookahead
    win_prob: float | None = None  # calibrated win probability; None when no calibration supports it
    opponent: list[tuple[str, float]] = field(default_factory=list)  # opponent's next action, most likely first
    policy: dict[str, float] = field(default_factory=dict)
    lookahead: list[tuple[str, float, float]] = field(default_factory=list)  # (label, prob, value)
    elapsed: float = 0.0
    # win_raw is the PPO critic mapped to [0, 1]: its rewards include HP / faint shaping and discounting, so it is not
    # a win probability. win_prob is it through a calibration for this source and policy
    # (pokesim.evaluation.calibration), or None ("not enough data") when there is none or it is not enabled
    win_raw: float | None = None
    win_kind: str = "policy value (uncalibrated)"
    status: dict = field(default_factory=dict)  # mode, fallback reason, failed branches, solver
    shadow: list[tuple[str, float, float]] = field(default_factory=list)  # planner advice in shadow mode
    evaluations: list[dict] = field(default_factory=list)  # comparison utilities, separate from action probabilities


def action_label(action: int, battle: Battle) -> str | None:
    """poke-env SinglesEnv action index -> lookahead-style label."""
    if action < 6:
        team = list(battle.team.values())
        return f"switch:{to_id(team[action].species)}" if action < len(team) else None
    if not 6 <= action < 14:
        return None
    moves = [m.id for m in action_moves(battle)]
    slot = (action - 6) % 4
    if slot >= len(moves):
        return None
    return f"move:{moves[slot]}" + (":mega" if action >= 10 else "")


# how the opponent's next action is predicted (pokesim.search.lookahead.opponent_mix); chosen on our own records
# with scripts/eval_records.py
# 166 decisions in 14 battles (2026-10-13): top-1 32% -> 42%, top-3 63% -> 69%, mean log-lik -2.65 -> -2.09
OPPONENT_MODEL = {"br_weight": 0.75, "tau": 0.7, "repeat": 0.3}
# share of the cloned human policy in the opponent prediction (scripts/eval_records.py score --bc-weight)
OPPONENT_POLICY_WEIGHT = 0.5
# planner defaults (pokesim.search.planner): rho guards against a wrong opponent prediction. "search": the worker
# settings the planner's worlds are evaluated with (services/sim-tools/lookahead.js): we play on with the expert
# continuation (the Baton Pass player's plans), the opponent with tactical and with disrupt (haze / phazing / Encore
# / status against set-up); a real end scored +-1 less a small tie-break for a slower win, a penalty for having no
# damage source left, one seed per policy, 5 turns on, no new pass after 2.5 s. "plans": on turns our active can set
# up, the expert plans (how far to stack) are played to the end of the battle first and the best one becomes our
# continuation (Advisor._choose_plan). Local A/B 2026-10-15 (100 battles each vs the heuristic bot and opp_baton):
# greedy/tactical 75%, with expert + plans 79% (not a significant difference); initial values, to be set on
# development data.
PLAN_CANDIDATES = ["expert", "expert@0", "expert@2", "expert@4", "expert@6"]
PLANNER_SEARCH = {
    "batonPolicy": True,
    "continuations": ["expert/tactical", "expert/disrupt"],
    "scoring": "terminal",
    "lostPathPenalty": 0.3,
    "winDecay": 0.0,
    "seeds": 1,
    "rollout": 5,
    "deadlineMs": 2500,
    "plans": {"candidates": PLAN_CANDIDATES, "opponents": ["tactical", "disrupt"], "seeds": 1, "maxTurns": 60},
}
# plan search (Advisor._choose_plan): another plan replaces the default only when it wins by this much more
PLAN_MARGIN = 0.1
PLANNER = {"mode": "shadow", "rho": 0.3, "cloned_weight": OPPONENT_POLICY_WEIGHT, "search": PLANNER_SEARCH}


class Advisor:
    def __init__(
        self,
        sets: list[dict],
        policy: str | None = None,
        lookahead: bool = True,
        seasons: tuple[str, ...] = ("M-6",),
        samples: int = 6,
        seeds: int = 2,
        rollout: int = 3,
        reward: str = "sparse",
        opponent_model: dict | None = None,
        opponent_policy: str | Path | None = "default",
        combine: dict | None = None,
        planner: dict | None = None,
        source: str | None = None,
        sample: bool = False,
        prior_bundle: dict | None = None,
        sampler: SetSampler | None = None,
        belief: bool = True,
        value_run: str | Path | None = None,
    ):
        """source: where the advice is used ("switch" / "showdown"), for the win calibration.
        belief: infer the opponent's sets from the damage and move order seen so far (pokesim.belief.sets) and
        let the lookahead draw them from that (needs the battle's log in advise)."""
        self.model = MaskablePPO.load(policy, device="cpu") if policy else None
        roster = [s["species"] for s in sets]
        self.extra = (
            TeamFeatures.for_model(self.model, roster, prior_bundle)
            if self.model
            else TeamFeatures(roster, prior_bundle=prior_bundle)
        )
        self.victory = REWARDS[reward]["victory_value"]
        self.combine = dict(COMBINE, **(combine or {}))
        self.sample = sample
        self.planner = dict(PLANNER, **planner) if planner else None
        if self.planner:
            self.planner["search"] = dict(PLANNER_SEARCH, **self.planner.get("search", {}))
        value_model = None
        if value_run is not None:
            from pokesim.value.runtime import ValueEnsemble

            if not lookahead or not self.planner or self.planner['mode'] not in ('on', 'shadow'):
                raise ValueError('value-run requires lookahead and planner on/shadow')
            value_model = ValueEnsemble(value_run)
            requested = (planner or {}).get('search', {})
            # priorVisits / rolloutPolicies: see services/sim-tools/terminal_search.js. The V_ref bootstrap of a
            # root cell keeps the weight of 4 rollouts, and the opponent's rollouts rotate over the scripted
            # disrupt / tactical foes and the same expert policy we use (symmetric rollouts).
            # workers: Node processes searching the same worlds on independent random streams
            # (POKESIM_SEARCH_WORKERS, default 4; about 400 MB each).
            self.planner['search'].update(plans=None, seeds=1, rollout=2, terminalSearch=True, decisionMs=15000,
                                          continuations=['expert/tactical', 'expert/disrupt'], tiebreak=0,
                                          lostPathPenalty=0, maxTurns=8, valueUtility='reference-material-v2',
                                          priorVisits=4, rootProgress=False, focusTopTwo=False,
                                          terminalReview=True, rootCounterplay=True,
                                          rolloutPolicies={'p1': ['expert'], 'p2': ['disrupt', 'tactical', 'expert',
                                                                                 'counterplay']},
                                          workers=int(os.environ.get('POKESIM_SEARCH_WORKERS', 4)))
            self.planner['search'].update({k: v for k, v in requested.items()
                                          if k not in ('plans', 'continuations', 'tiebreak', 'lostPathPenalty')})
            self.planner['search']['value_model'] = value_model.version
        from pokesim.evaluation.calibration import Calibrations, continuation_context
        from pokesim.runtime.bundle import file_hash

        self.calibrations = Calibrations()
        self.calibration_context = continuation_context(
            sets,
            self.planner if self.planner and self.planner["mode"] == "on" else None,
            self.combine,
            reward,
            lookahead,
            sample,
        )
        self.calibration = (
            self.calibrations.get(source, file_hash(policy), self.calibration_context) if policy else None
        )
        model = OPPONENT_MODEL if opponent_model is None else opponent_model
        search = self.planner["search"] if self.planner and self.planner["mode"] == "on" else None
        self.engine = (
            Lookahead(
                sampler if sampler is not None else SetSampler(list(seasons)),
                samples,
                seeds,
                rollout,
                model,
                search=search,
                value_model=value_model,
            )
            if lookahead
            else None
        )
        self.beliefs = None
        if self.engine and sampler is None:
            from pokesim.belief.team_catalog import TeamCatalog

            self.engine.sampler.catalog = TeamCatalog.load(self.engine.sampler.base_id)
        self.last_plan: dict | None = None  # the plan search of the last _search (see _choose_plan)
        if belief and self.engine:
            from pokesim.belief.sets import Beliefs

            self.beliefs = Beliefs(sets, self.engine.sampler)
        if self.engine and getattr(self.engine.value_model, 'branch_policy', None):
            self.planner['search'].setdefault('policyNets', {})['ours'] = self.engine.value_model.branch_policy
            if self.engine.value_model.version['reference'].get('policy_sha'):
                self.planner['search'].setdefault('rolloutMix', {})['ours'] = 1.
                self.planner['search'].setdefault('rolloutTemperature', {})['ours'] = 0.
                self.planner['search']['rolloutNetTurns'] = 200
        base = self.engine.sampler.base_id if self.engine else (lambda s: s)
        self.own_sets = {base(to_id(s["species"])): s for s in sets}
        self.opponent_policy = None
        self.opponent_policy_version: dict | None = None  # which cloned policy predicted the opponent (records)
        if isinstance(opponent_policy, str) and opponent_policy.startswith("net:"):
            # A learned public policy (pokesim.learn) run inside the worker: tree priors, rollouts and the root
            # prediction's learned share (planner.predict). The cloned PPO policy is not loaded.
            net = Path(opponent_policy[4:])
            if not net.exists():
                raise FileNotFoundError(f"opponent policy net not found: {net}")
            if not self.planner or not self.planner["search"].get("terminalSearch"):
                raise ValueError("an opponent policy net needs the terminal search (planner on with a value run)")
            self.planner["search"].setdefault("policyNets", {})["opp"] = str(net.resolve())
            self.opponent_policy_version = {"path": net.name, "sha": file_hash(net), "kind": "net"}
        elif opponent_policy:
            from pokesim.live.opponent import DEFAULT_POLICY, OpponentPolicy

            path = DEFAULT_POLICY if opponent_policy == "default" else Path(opponent_policy)
            if Path(path).exists():
                self.opponent_policy = OpponentPolicy(path)
                self.opponent_policy_version = {"path": Path(path).name, "sha": file_hash(path)}
            elif opponent_policy != "default":
                raise FileNotFoundError(f"opponent policy not found: {path}")

    def close(self) -> None:
        if self.engine:
            self.engine.close()
        if self.beliefs:
            self.beliefs.close()

    def _policy(self, battle: Battle) -> tuple[dict[str, float], float | None]:
        if self.model is None:
            return {}, None
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        if not mask.any():
            return {}, None
        obs = np.concatenate([embed(battle), self.extra(battle)])
        probs = action_probs(self.model, obs, mask)
        out = {}
        for a in np.flatnonzero(mask):
            label = action_label(int(a), battle)
            if label:
                out[label] = out.get(label, 0.0) + float(probs[a])
        with torch.no_grad():
            value = float(self.model.policy.predict_values(self.model.policy.obs_to_tensor(obs)[0])[0, 0])
        return out, float(np.clip((value / self.victory + 1) / 2, 0, 1))

    def advise(
        self, battle: Battle, last_move: str | None = None, log: list[str] | None = None, our_role: str = "p1"
    ) -> Suggestion:
        """last_move: the opponent active's previous move ("move:<id>"), see BattleSession.last_move.
        log: the battle's protocol lines so far (BattleSession.log), for the cloned opponent policy."""
        start = time.perf_counter()
        policy, win_raw = self._policy(battle)
        # This calibration was validated only at the first normal decision, not on arbitrary later positions.
        supported_point = battle.turn == 1 and not battle.force_switch
        win = self.calibration.apply(win_raw) if self.calibration and supported_point else None
        ranked, opponent, shadow = [], [], []
        mode = self.planner["mode"] if self.planner else "legacy"
        status: dict = {"mode": mode}
        status["calibration"] = {"supported_point": supported_point, "context": self.calibration_context}
        cloned = []
        if self.opponent_policy is not None and log and battle.opponent_active_pokemon is not None:
            cloned = self.opponent_policy.predict(log, our_role)
        opp_active = battle.opponent_active_pokemon
        # After a double KO both actives are fainted and both sides pick a replacement at once; the simulator
        # rebuilds that as two forced switches, so ours is still searched. Without a policy (--policy none) it
        # would otherwise be a random switch, and the quality gate stops collection. (A forced switch of a live
        # active while the opponent's is fainted, a U-turn KO, is not searched: the rebuilt root has no switch.)
        ours = battle.active_pokemon
        double_ko = battle.force_switch and (ours is None or ours.fainted)
        can_search = (
            self.engine
            and opp_active is not None
            and (not opp_active.fainted or double_ko)
            and (ours is not None or battle.force_switch)
        )
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)

        def legal(advice):
            return [
                (lab, p, v)
                for lab, p, v in advice.ranked()
                if (a := label_to_action(lab, battle)) is not None and mask[a]
            ]

        belief = None
        if self.beliefs is not None and log and opp_active is not None:
            belief_start = time.perf_counter()
            try:
                belief = self.beliefs.get(battle.battle_tag, log, our_role)
                status["belief"] = {"observations": len(belief.observations),
                                    "speed_observations": sum(o.kind == 'speed' for o in belief.observations),
                                    "speed_support": belief.speed_support(opp_active.species, *revealed(opp_active)),
                                    "active": belief.summary(opp_active.species, *revealed(opp_active))}
            except Exception as e:  # noqa: BLE001 - without the belief the lookahead draws from the prior
                belief = None
                status["belief"] = {"error": repr(e)[:200]}
            status["belief"]["elapsed_ms"] = 1000 * (time.perf_counter() - belief_start)
        if can_search:
            try:
                cache: dict = {}
                settings = None
                if mode == 'on' and self.planner['search'].get('terminalSearch'):
                    from pokesim.search.budget import decision_budget

                    # Inference from the observed log is part of this decision's time budget.
                    settings = dict(self.planner['search'])
                    allowance = decision_budget(settings.get('decisionMs', 8500), log)
                    status['decision_budget_ms'] = allowance
                    settings['decisionMs'] = max(100, allowance -
                                                 int(1000 * (time.perf_counter() - start)))
                advice = self._search(battle, last_move, dict(cloned), planned=mode == "on", cache=cache,
                                      belief=belief, log=log, our_role=our_role, settings=settings)
                ranked = legal(advice)
                opponent = sorted(advice.opponent.items(), key=lambda kv: -kv[1])
                status["search"] = advice.status
                if 'root_evidence' not in advice.status and not advice.status.get('terminal_search', {}).get(
                        'immediate_wins'):
                    # The fixed-depth/legacy API also needs actual scores, not just selection probabilities.
                    risk = self.planner['rho'] if mode == 'on' else 0.
                    advice.status['root_evidence'] = [
                        {'label': label, 'expected': float(advice.value[i]),
                         'worst': float(advice.worst[i]) if advice.worst is not None else None,
                         'utility': float((1 - risk) * advice.value[i] + risk * advice.worst[i])
                         if advice.worst is not None else float(advice.value[i]), 'trials': 0}
                        for i, label in enumerate(advice.labels)]
                    advice.status['score_kind'] = 'uncalibrated-fixed-depth-utility'
                if mode == "shadow":
                    try:
                        planned = self._search(
                            battle, last_move, dict(cloned), planned=True, settings=self.planner["search"],
                            belief=belief, log=log, our_role=our_role,
                        )
                        shadow = legal(planned)
                        status["shadow"] = planned.status
                    except Exception as e:  # noqa: BLE001 - observation-only failures must preserve live advice
                        status["shadow"] = {"error": repr(e)[:200]}
            except Exception as e:  # noqa: BLE001 - a failed search must not lose the turn: the policy decides
                ranked = []
                status["fallback"] = f"search failed: {e!r}"[:200]
        if mode == "on" and ranked:  # the plan decides; the policy is only the fallback
            total = sum(p for _, p, _ in ranked)
            actions = (
                sorted(((lab, p / total) for lab, p, _ in ranked), key=lambda kv: -kv[1])
                if total > 0
                else combine({}, ranked, **self.combine)
            )
        else:
            actions = combine(policy, ranked, **self.combine)
        if mode != "on" and cloned:
            opponent = mix_opponent(opponent, cloned)
        if not actions and not status.get("fallback") and not policy:
            status["fallback"] = "no advice"
        try:
            status["win_path"] = win_path(battle).as_dict()
        except Exception as e:  # noqa: BLE001 - a diagnostic only
            status["win_path"] = {"error": repr(e)[:120]}
        return Suggestion(
            actions,
            win,
            opponent,
            policy,
            ranked,
            time.perf_counter() - start,
            status=status,
            shadow=shadow,
            win_raw=win_raw,
            win_kind="calibrated" if win is not None else "policy value (uncalibrated)",
            evaluations=action_evaluations(status, actions[0][0] if actions else None),
        )

    def _search(
        self,
        battle: Battle,
        last_move: str | None,
        cloned: dict[str, float],
        planned: bool,
        cache: dict | None = None,
        settings: dict | None = None,
        belief=None,
        log=None,
        our_role='p1',
    ):
        """The legacy aggregate or the planner (planned) on the lookahead's worlds; cache keeps the worlds of one
        position (by incoming Pokémon). settings: worker settings for these queries (shadow mode evaluates the
        planner's worlds with its own settings, so it measures the new method as a whole)."""
        from pokesim.search.planner import plan, predict

        engine = self.engine
        cache = {} if cache is None else cache
        self.last_plan = None

        def solve(b, own, last, incoming=None):
            key = None if incoming is None else incoming.species
            if key not in cache:
                extra = {}
                if planned and getattr(engine, 'value_model', None) is not None:
                    # The same opponent prediction the planner uses, so the search examines replies by it.
                    def predictor(payoff, labels, prior=None, last=last):
                        return predict(payoff, labels, engine.opponent_model, last, cloned or None,
                                       self.planner["cloned_weight"], prior)

                    extra = {'value': True, 'log': log, 'our_role': our_role, 'predictor': predictor}
                cache[key] = engine.query(b, own, incoming, settings, belief=belief, **extra)
            if planned:
                advice = plan(
                    cache[key],
                    engine.opponent_model,
                    last,
                    self.planner["rho"],
                    cloned or None,
                    self.planner["cloned_weight"],
                    pure=not self.sample or self.planner.get("near") is not None,
                    classes=action_classes(b) if self.planner.get("rho_by_class") else None,
                    rho_by_class=self.planner.get("rho_by_class"),
                    near=self.planner.get("near"),
                    tie_margin=self.planner.get("tie_margin"),
                )
                values = [res['value_status'] for res in cache[key] if 'value_status' in res]
                if values:
                    advice.status['value'] = {'model': values[0]['model'],
                                              **{k: sum(v[k] for v in values)
                                                 for k in ('terminal', 'learned', 'failed')},
                                              'inference_ms': max(v['inference_ms'] for v in values)}
                return advice
            return aggregate(cache[key], engine.opponent_model, last)

        if (planned and getattr(engine, 'value_model', None) is not None and
                (settings or self.planner['search']).get('terminalSearch')):
            # One joint query includes every forced-switch candidate and shares one decision budget.
            return solve(battle, self.own_sets, last_move)
        if battle.force_switch or battle.active_pokemon is None or battle.active_pokemon.fainted:
            return engine.advise_switch(battle, self.own_sets, last_move, solve)
        current = settings if settings is not None else (self.planner or {}).get("search", {})
        if planned and current.get("plans") and can_set_up(battle.active_pokemon):
            chosen = self._choose_plan(battle, current, belief)
            if chosen is not None:
                settings = with_plan(current, chosen)
        advice = solve(battle, self.own_sets, last_move)
        if self.last_plan is not None:
            advice.status["plans"] = self.last_plan
        return advice

    def _choose_plan(self, battle: Battle, settings: dict, belief) -> str | None:
        """Our plan for this position (how far to stack, when to heal): the candidate with the best result played
        to the end, over all sampled worlds together (one choice for every world: we cannot see which one is real).
        The first candidate is the default: another replaces it only when it scores higher by plans["margin"]
        (default PLAN_MARGIN), so a few games' noise does not change the plan. None when the search fails."""
        self.last_plan = None
        try:
            reply = self.engine.plans(battle, self.own_sets, settings["plans"], belief)
        except Exception as e:  # noqa: BLE001 - the default plan still plays
            self.last_plan = {"error": repr(e)[:200]}
            return None
        scores = {name: r["score"] for name, r in reply["plans"].items() if r["score"] is not None}
        default = settings["plans"]["candidates"][0]
        best = max(scores, key=scores.get) if scores else None
        margin = settings["plans"].get("margin", PLAN_MARGIN)
        chosen = best if best is not None and scores[best] > scores.get(default, -1.0) + margin else default
        self.last_plan = {"chosen": chosen, "scores": {k: round(v, 3) for k, v in scores.items()},
                          "unfinished": {k: r["unfinished"] for k, r in reply["plans"].items() if r["unfinished"]},
                          "cycles": reply["cycles"], "ms": reply["elapsedMs"]}
        return chosen


def action_evaluations(status: dict, selected: str | None = None) -> list[dict]:
    """Scores compare moves; the probability of selecting a move is a separate quantity."""
    search = status.get('search', {})
    rows = [dict(row) for row in search.get('root_evidence', [])]
    if rows:
        best = max(row['utility'] for row in rows)
        for row in rows:
            row['gap'] = best - row['utility']
            row['selected'] = row['label'] == selected
        rows.sort(key=lambda row: (not row['selected'], -row['utility']))
    elif search.get('solver') == 'sampled-world-immediate-win':
        rows = [{'label': label, 'utility': 1., 'worst': 1., 'expected': 1., 'gap': 0., 'certified': True,
                 'selected': label == selected}
                for label in search['terminal_search']['immediate_wins']]
    excluded = search.get('terminal_search', {}).get('action_filter', {}).get('excluded', [])
    return rows + [dict(row, utility=None, worst=None, expected=None, gap=None, excluded=True) for row in excluded]


def action_classes(battle: Battle) -> dict[str, str]:
    """'setup' for the active's moves that raise its own stats or Baton Pass (and their Mega variants)."""
    from poke_env.battle import Target

    mon = battle.active_pokemon
    if mon is None:
        return {}
    out = {}
    for m in mon.moves.values():
        if m.id == "batonpass" or (m.target == Target.SELF and m.boosts and any(v > 0 for v in m.boosts.values())):
            out[f"move:{m.id}"] = "setup"
            out[f"move:{m.id}:mega"] = "setup"
    return out


def can_set_up(mon) -> bool:
    """Whether the active has a move that raises its own stats or Baton Pass: the positions plans differ in."""
    from poke_env.battle import Target

    return any(m.id == "batonpass" or (m.target == Target.SELF and m.boosts and any(v > 0 for v in m.boosts.values()))
               for m in mon.moves.values())


def with_plan(settings: dict, plan: str) -> dict:
    """The worker settings with our continuation "expert" (alone or as "expert/<theirs>") replaced by the plan."""
    def ours(name: str) -> str:
        mine, sep, theirs = name.partition("/")
        return (plan if mine == "expert" else mine) + sep + theirs
    return dict(settings, continuations=[ours(c) for c in settings.get("continuations", [])])


def combine(
    policy: dict[str, float],
    ranked: list[tuple[str, float, float]],
    mode: str = "product",
    weight: float = POLICY_WEIGHT,
    temperature: float = 1.0,
    tau: float = 0.5,
    margin: float | None = None,
    defer: float | None = None,
) -> list[tuple[str, float]]:
    """Our action distribution from the policy (label -> prob) and the lookahead ((label, eq. prob, value)).

    linear:  weight * policy + (1 - weight) * lookahead equilibrium mix.
    product: policy^(weight / temperature) * lookahead^(1 - weight), where the lookahead's distribution is the mean of
             its equilibrium mix and softmax(value / tau); an action needs support from both. Actions whose
             lookahead value is more than `margin` below the best (a win counts 3, a Pokemon's HP 1) are dropped.
    defer: when the lookahead's equilibrium puts at least this much on one action, its mix alone (the policy cannot
    see unrevealed moves or Mega Evolution; the search samples them).
    With only one source, that source alone.
    If the lookahead's mix gives no weight to any legal action (it was spent on options the battle does not allow),
    its mix is replaced by softmax(value / tau) over the legal ones."""
    if ranked and sum(p for _, p, _ in ranked) <= 0:
        values = np.array([v for _, _, v in ranked])
        soft = np.exp((values - values.max()) / tau)
        ranked = [(lab, float(q), v) for (lab, _, v), q in zip(ranked, soft / soft.sum())]
    if not ranked:
        out = dict(policy)
    elif not policy or (defer is not None and max(p for _, p, _ in ranked) >= defer * sum(p for _, p, _ in ranked)):
        total = sum(p for _, p, _ in ranked) or 1.0
        out = {lab: p / total for lab, p, _ in ranked}
    elif mode == "linear":
        total = sum(p for _, p, _ in ranked) or 1.0
        out = {}
        for lab, p, _ in ranked:
            out[lab] = out.get(lab, 0.0) + (1 - weight) * p / total
        for lab, p in policy.items():
            out[lab] = out.get(lab, 0.0) + weight * p
    else:
        labels = [lab for lab, _, _ in ranked]
        values = np.array([v for _, _, v in ranked])
        eq = np.array([p for _, p, _ in ranked])
        eq = eq / eq.sum() if eq.sum() > 0 else np.full(len(eq), 1 / len(eq))
        soft = np.exp((values - values.max()) / tau)
        look = 0.5 * eq + 0.5 * soft / soft.sum()
        pol = np.array([policy.get(lab, 0.0) for lab in labels]) + 1e-4
        score = (weight / temperature) * np.log(pol) + (1 - weight) * np.log(look + 1e-4)
        if margin is not None:
            score[values < values.max() - margin] = -np.inf
        prob = np.exp(score - score.max())
        out = dict(zip(labels, (prob / prob.sum()).tolist()))
    return sorted(out.items(), key=lambda kv: -kv[1])


def mix_opponent(
    search: list[tuple[str, float]], cloned: list[tuple[str, float]], w: float = OPPONENT_POLICY_WEIGHT
) -> list[tuple[str, float]]:
    """Share w for the cloned policy (all of it when the search has nothing); Mega variants merged."""
    if not cloned:
        return search
    w = w if search else 1.0
    out: dict[str, float] = {}
    for lab, p in search:
        key = ":".join(lab.split(":")[:2])
        out[key] = out.get(key, 0.0) + (1 - w) * p
    for lab, p in cloned:
        out[lab] = out.get(lab, 0.0) + w * p
    return sorted(out.items(), key=lambda kv: -kv[1])
