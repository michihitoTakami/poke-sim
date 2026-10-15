"""Compare the planner in shadow mode with the moves we actually played (ladder records, --planner shadow).

summary: from the records alone. Per decision: what we played (the legacy advice: policy x lookahead), what the
planner would have played (its top action), and the kind of disagreement (Mega timing, attack vs status move,
move vs switch, ...). Results cannot say which was better: a battle's outcome follows the moves played, and the
battles where the two disagree more are not comparable with the others (harder positions disagree more).

referee: re-scores normal decisions where the two disagree with a heavier search than either used live: more
sampled worlds, more seeds, longer rollouts, both continuation policies, no deadline, fresh sampler seeds and the
offline prior bundle (fitted on train battles only, so the opponent's sets later in the same battle cannot leak in).
Per position, both candidate actions get
  value  expected score against the predicted opponent (the live opponent model and cloned policy), mean over worlds
  worst  each world's worst case over the opponent's options, mean over worlds
  delta = value(planner's action) - value(the action played), its standard error over worlds, and the same delta
          under each continuation policy on its own.
A verdict is "planner" / "played" only when |delta| exceeds two standard errors and every continuation policy
agrees on the sign; otherwise "unclear". The referee is the planner's own simulator and continuation policies with
more compute: it measures whether the planner's choice survives a closer look, not the truth. A decision to act on
the planner still needs the A/B on battles played after the ones the referee looked at.

  uv run python -m pokesim.evaluation.shadow summary [--since 20261015-033700]
  uv run python -m pokesim.evaluation.shadow referee --limit 200    # resumable; cached per (battle, turn)
  uv run python -m pokesim.evaluation.shadow report                 # summary + referee results
"""
import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pokesim.config import PROCESSED_DIR, ROOT
from pokesim.evaluation.metrics import bootstrap, wilson
from pokesim.rl.myteam import to_id
from pokesim.team.roles import move_data

LADDER = ROOT / "data/raw/ladder"
REFEREE_CACHE = PROCESSED_DIR / "shadow_referee.jsonl"
# heavier than live (shadow: 6 worlds, 1 seed per continuation, 5 turns on, 2.5 s deadline); no lost-path penalty:
# it is one of the planner's own heuristics, and longer rollouts reach more real ends instead
REFEREE = {"samples": 12, "search": {"continuations": ["greedy", "tactical"], "scoring": "terminal",
                                     "lostPathPenalty": 0.0, "winDecay": 0.0, "seeds": 2, "rollout": 8}}
MARGIN_SE = 2.0  # a verdict needs |delta| above this many standard errors


# --- records ---------------------------------------------------------------------------------------------------
def records(since: str | None = None, until: str | None = None, root: Path = LADDER) -> list[tuple[str, dict]]:
    """Ladder records with shadow advice, oldest first (by file name: the start time)."""
    out = []
    for p in sorted(root.glob("*.json")):
        if (since and p.stem < since) or (until and p.stem >= until):
            continue
        rec = json.loads(p.read_text())
        if any(a.get("shadow") for a in rec.get("advice", [])):
            out.append((p.stem, rec))
    return out


def move_kind(label: str) -> str:
    kind, _, rest = label.partition(":")
    if kind != "move":
        return kind
    return "attack" if move_data(rest.split(":")[0]).get("category") in ("Physical", "Special") else "status"


def disagreement(played: str, planned: str) -> str:
    """The kind of a disagreement, from the action played to the planner's."""
    if played == planned:
        return "agree"
    a, b = played.split(":"), planned.split(":")
    if a[0] == b[0] == "move" and a[1] == b[1]:
        return "mega timing"
    return f"{move_kind(played)} -> {move_kind(planned)}"


@dataclass
class Decision:
    battle: str
    turn: int
    force: bool
    result: str | None
    policy: str | None
    commit: str | None
    played: str
    planned: str | None  # None: no shadow advice (search failed or not run)
    kind: str
    secs: float | None
    planner_gap: float | None = None  # planner's own value: its action minus the one played (None: not saved)
    error: str | None = None


def decisions(recs: list[tuple[str, dict]]) -> list[Decision]:
    out = []
    for stem, rec in recs:
        commit = ((rec.get("versions") or {}).get("code") or {}).get("commit")
        for a in rec.get("advice", []):
            played = a.get("chosen")
            if not played:
                continue
            shadow = a.get("shadow") or []
            planned = max(shadow, key=lambda s: s[1])[0] if shadow else None
            values = {lab: v for lab, _, v in shadow}
            gap = values[planned] - values[played] if planned and played in values else None
            error = ((a.get("status") or {}).get("shadow") or {}).get("error")
            out.append(Decision(stem, a.get("turn"), bool(a.get("force")), rec.get("result"), rec.get("policy"),
                                commit, played, planned, disagreement(played, planned) if planned else "no shadow",
                                a.get("secs"), gap, error))
    return out


def rate(k: int, n: int) -> str:
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}, 95% {lo:.0%}-{hi:.0%})"


def summary(ds: list[Decision]) -> dict:
    with_shadow = [d for d in ds if d.planned]
    battles = {d.battle for d in ds}
    out = {"battles": len(battles), "decisions": len(ds), "with_shadow": len(with_shadow),
           "shadow_errors": sum(d.error is not None for d in ds)}
    for name, sel in (("normal", [d for d in with_shadow if not d.force]),
                      ("forced switch", [d for d in with_shadow if d.force])):
        k = sum(d.planned != d.played for d in sel)
        out[f"disagree ({name})"] = rate(k, len(sel))
    dis = [d for d in with_shadow if d.planned != d.played]
    out["kinds"] = dict(Counter(d.kind for d in dis).most_common())
    by_turn = defaultdict(lambda: [0, 0])
    for d in with_shadow:
        if not d.force:
            t = "1" if d.turn == 1 else "2-3" if d.turn <= 3 else "4-6" if d.turn <= 6 else "7+"
            by_turn[t][0] += d.planned != d.played
            by_turn[t][1] += 1
    out["disagree by turn (normal)"] = {t: rate(*by_turn[t]) for t in ("1", "2-3", "4-6", "7+") if t in by_turn}
    by_policy = defaultdict(lambda: [0, 0])
    for d in with_shadow:
        by_policy[d.policy][0] += d.planned != d.played
        by_policy[d.policy][1] += 1
    out["disagree by policy"] = {p: rate(*v) for p, v in sorted(by_policy.items(), key=str)}
    gaps = [d.planner_gap for d in dis if d.planner_gap is not None]
    if gaps:
        out["planner's own gap (its value of its action minus the one played)"] = {
            "n": len(gaps), "median": round(float(np.median(gaps)), 3), "p90": round(float(np.quantile(gaps, 0.9)), 3),
            "not saved (played action outside the saved top 5)": len(dis) - len(gaps)}
    secs = sorted(d.secs for d in ds if d.secs is not None)
    if secs:
        out["seconds per decision (legacy + shadow)"] = {"median": secs[len(secs) // 2],
                                                         "p95": secs[int(0.95 * (len(secs) - 1))], "max": secs[-1]}
    # descriptive only: battles that disagree more are different battles, not a comparison of the two methods
    share = defaultdict(lambda: [0, 0])
    for d in with_shadow:
        share[d.battle][0] += d.planned != d.played
        share[d.battle][1] += 1
    results = {d.battle: d.result for d in ds}
    rows = {"low": [0, 0], "high": [0, 0]}
    med = float(np.median([k / n for k, n in share.values()])) if share else 0.0
    for b, (k, n) in share.items():
        if results[b] in ("win", "loss"):
            g = "high" if k / n > med else "low"
            rows[g][0] += results[b] == "win"
            rows[g][1] += 1
    out[f"win rate by disagreement share (descriptive, split at median {med:.0%})"] = {
        g: rate(*v) for g, v in rows.items()}
    return out


# --- referee ---------------------------------------------------------------------------------------------------
@dataclass
class Verdict:
    battle: str
    turn: int
    kind: str
    played: str
    planned: str
    status: str = "ok"  # ok | skipped: <reason> | failed: <reason>
    value: dict = field(default_factory=dict)  # label -> expected score against the prediction
    worst: dict = field(default_factory=dict)  # label -> mean worst case
    delta: float | None = None
    se: float | None = None
    worst_delta: float | None = None
    by_continuation: dict = field(default_factory=dict)  # name -> delta
    resolved: float | None = None  # share of rollouts that reached a real end (both actions, all worlds)
    worlds: int = 0
    verdict: str = "unclear"
    secs: float = 0.0
    settings: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        return self.battle, self.turn


def judge(delta: float, se: float, by: dict[str, float]) -> str:
    if abs(delta) <= MARGIN_SE * se:
        return "unclear"
    side = np.sign(delta)
    if by and any(np.sign(v) != side for v in by.values()):
        return "unclear"
    return "planner" if side > 0 else "played"


def score_worlds(results: list[dict], played: str, planned: str, opponent_model: dict, last_move: str | None,
                 cloned: dict[str, float] | None, cloned_weight: float) -> dict:
    """Both actions in each world: expected score against the predicted opponent, worst case, per continuation."""
    from pokesim.search.lookahead import clean
    from pokesim.search.planner import matrix, predict

    per_world, worst, by, resolved = [], [], defaultdict(list), []
    labels = results[0]["p1"]
    names = sorted({n for res in results for n in res.get("byContinuation", {})})
    for res in results:
        try:
            payoff, opp, _ = clean(res)
        except ValueError:
            continue
        idx = [res["p1"].index(lab) for lab in labels]
        payoff = payoff[idx]
        q = predict(payoff, opp, opponent_model, last_move, cloned, cloned_weight)
        rows = [labels.index(played), labels.index(planned)]
        per_world.append(payoff[rows] @ q)
        worst.append(payoff[rows].min(axis=1))
        cols = [res["p2"].index(lab) for lab in opp]
        for name in names:  # a world without one (as in planner.diagnose) counts its payoff
            m = res.get("byContinuation", {}).get(name)
            by[name].append((payoff if m is None else matrix(m, idx, cols, payoff))[rows] @ q)
        if "resolved" in res:
            resolved.append(float(matrix(res["resolved"], idx, cols, np.zeros_like(payoff))[rows].mean()))
    if not per_world:
        raise ValueError("every sampled world failed")
    v = np.array(per_world)  # worlds x [played, planned]
    d = v[:, 1] - v[:, 0]
    w = np.array(worst)
    return {"value": {played: float(v[:, 0].mean()), planned: float(v[:, 1].mean())},
            "worst": {played: float(w[:, 0].mean()), planned: float(w[:, 1].mean())},
            "delta": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf"),
            "worst_delta": float(w[:, 1].mean() - w[:, 0].mean()),
            "by_continuation": {n: float(np.mean([x[1] - x[0] for x in xs])) for n, xs in by.items()},
            "resolved": float(np.mean(resolved)) if resolved else None, "worlds": len(per_world)}


def position(rec: dict, sets: list[dict], turn: int):
    """The battle rebuilt from the record's protocol log at the start of `turn` (our side is p1 in records)."""
    from pokesim.live.session import BattleSession

    session = BattleSession(sets)
    session.preview(rec["preview"]["species"])
    session.select([next(i for i, s in enumerate(sets) if to_id(s["species"]) == to_id(x)) for x in rec["lineup"]])
    log = rec["log"]
    start = next(i for i, line in enumerate(log) if line.startswith("|teampreview")) + 1
    end = log.index(f"|turn|{turn}")
    # server replies to a rejected choice ("|error|[Invalid choice] ...") are not battle events
    session.apply([line for line in log[start:end + 1] if not line.startswith("|error|")])
    return session, session.decision()


class Referee:
    def __init__(self, sets: list[dict], prior_bundle: Path, seed: int = 0, cloned: bool = True,
                 samples: int = REFEREE["samples"], search: dict | None = None):
        from pokesim.data.prior_bundle import load_bundle, sampler
        from pokesim.live.advisor import OPPONENT_MODEL, OPPONENT_POLICY_WEIGHT
        from pokesim.search.lookahead import Lookahead

        bundle = load_bundle(prior_bundle)
        self.sets = sets
        self.search = dict(REFEREE["search"], **(search or {}))
        self.engine = Lookahead(sampler(bundle, seed), samples, opponent_model=OPPONENT_MODEL, search=self.search)
        self.own_sets = {self.engine.sampler.base_id(to_id(s["species"])): s for s in sets}
        self.cloned_weight = OPPONENT_POLICY_WEIGHT
        self.opponent_policy = None
        if cloned:
            from pokesim.live.opponent import DEFAULT_POLICY, OpponentPolicy

            if Path(DEFAULT_POLICY).exists():
                self.opponent_policy = OpponentPolicy(DEFAULT_POLICY, sampler=sampler(bundle, seed))
        self.settings = {"samples": samples, "search": self.search, "seed": seed, "cloned": self.opponent_policy
                         is not None, "prior_bundle": str(prior_bundle)}

    def close(self) -> None:
        self.engine.close()

    def __call__(self, rec: dict, d: Decision) -> Verdict:
        v = Verdict(d.battle, d.turn, d.kind, d.played, d.planned, settings=self.settings)
        start = time.perf_counter()
        try:
            session, battle = position(rec, self.sets, d.turn)
        except Exception as e:  # noqa: BLE001 - a log the session cannot follow: skip the position, keep going
            v.status = f"skipped: cannot rebuild ({e!r})"[:200]
            return v
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            v.status = "skipped: no active Pokemon"
            return v
        try:
            results = self.engine.query(battle, self.own_sets)
            labels = results[0]["p1"]
            missing = [lab for lab in (d.played, d.planned) if lab not in labels]
            if missing:
                v.status = f"skipped: not an option in the rebuilt position: {missing}"
                return v
            cloned = dict(self.opponent_policy.predict(session.log)) if self.opponent_policy else None
            scores = score_worlds(results, d.played, d.planned, self.engine.opponent_model, session.last_move(),
                                  cloned or None, self.cloned_weight)
        except Exception as e:  # noqa: BLE001 - one bad position must not stop the run
            v.status = f"failed: {e!r}"[:200]
            return v
        for k, val in scores.items():
            setattr(v, k, val)
        v.verdict = judge(v.delta, v.se, v.by_continuation)
        v.secs = round(time.perf_counter() - start, 2)
        return v


def load_verdicts(path: Path) -> dict[tuple[str, int], Verdict]:
    if not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        v = Verdict(**json.loads(line))
        out[v.key] = v
    return out


def referee_candidates(ds: list[Decision]) -> list[Decision]:
    """Normal decisions where the planner disagreed (forced switches are a different search; first one per turn)."""
    seen, out = set(), []
    for d in ds:
        if d.force or not d.planned or d.planned == d.played or (d.battle, d.turn) in seen:
            continue
        seen.add((d.battle, d.turn))
        out.append(d)
    return out


def referee_summary(vs: list[Verdict]) -> dict:
    ok = [v for v in vs if v.status == "ok"]
    out = {"positions": len(vs), "scored": len(ok),
           "skipped / failed": dict(Counter(v.status.split(":")[0] for v in vs if v.status != "ok"))}
    if not ok:
        return out
    groups = [v.battle for v in ok]
    delta = np.array([v.delta for v in ok])
    out["mean delta (planner - played)"] = round(float(delta.mean()), 3)
    out["95% interval (battles resampled)"] = [round(x, 3) for x in bootstrap(lambda i: float(delta[i].mean()),
                                                                            groups)]
    worst = np.array([v.worst_delta for v in ok])
    out["mean worst-case delta"] = round(float(worst.mean()), 3)
    out["verdicts"] = dict(Counter(v.verdict for v in ok))
    names = sorted({n for v in ok for n in v.by_continuation})
    out["mean delta by continuation"] = {n: round(float(np.mean([v.by_continuation[n] for v in ok
                                                                if n in v.by_continuation])), 3) for n in names}
    res = [v.resolved for v in ok if v.resolved is not None]
    if res:
        out["rollouts that reached a real end"] = f"{np.mean(res):.0%}"
    by_kind = defaultdict(list)
    for v in ok:
        by_kind[v.kind].append(v)
    out["by kind"] = {k: {"n": len(xs), "mean delta": round(float(np.mean([x.delta for x in xs])), 3),
                          "planner": sum(x.verdict == "planner" for x in xs),
                          "played": sum(x.verdict == "played" for x in xs)}
                      for k, xs in sorted(by_kind.items(), key=lambda kv: -len(kv[1]))}
    return out


def worst_cases(vs: list[Verdict], n: int = 10) -> dict[str, list[dict]]:
    """Clear verdicts with the largest margins, both ways: positions worth a human look."""
    ok = [v for v in vs if v.status == "ok"]

    def row(v: Verdict) -> dict:
        return {"battle": v.battle, "turn": v.turn, "played": v.played, "planner": v.planned,
                "delta": round(v.delta, 3), "se": round(v.se, 3), "by_continuation":
                {k: round(x, 3) for k, x in v.by_continuation.items()}}

    planner = sorted((v for v in ok if v.verdict == "planner"), key=lambda v: -v.delta)[:n]
    played = sorted((v for v in ok if v.verdict == "played"), key=lambda v: v.delta)[:n]
    return {"planner clearly better": [row(v) for v in planner], "played clearly better": [row(v) for v in played]}


# --- CLI -------------------------------------------------------------------------------------------------------
def print_dict(title: str, d: dict) -> None:
    print(f"== {title}")
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")


def main() -> None:
    from pokesim.data.prior_bundle import DEFAULT

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "referee", "report"])
    ap.add_argument("--since", help="first record (file name prefix, e.g. 20261015-033700)")
    ap.add_argument("--until", help="records before this (file name prefix)")
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--cache", type=Path, default=REFEREE_CACHE)
    ap.add_argument("--limit", type=int, help="referee at most this many new positions (random order)")
    ap.add_argument("--seed", type=int, default=0, help="position order and the referee's sampler")
    ap.add_argument("--prior-bundle", type=Path, default=DEFAULT)
    ap.add_argument("--no-cloned", action="store_true", help="opponent prediction without the cloned policy")
    ap.add_argument("--nice", type=int, default=10, help="lower priority, for a collector running alongside")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    recs = records(args.since, args.until)
    ds = decisions(recs)
    if args.cmd in ("summary", "report"):
        s = summary(ds)
        if args.cmd == "report":
            vs = [v for v in load_verdicts(args.cache).values() if v.battle in {stem for stem, _ in recs}]
            s = {"summary": s, "referee": referee_summary(vs), "cases": worst_cases(vs)}
        if args.json:
            print(json.dumps(s, indent=1, ensure_ascii=False))
        elif args.cmd == "summary":
            print_dict(f"shadow vs played ({len(recs)} battles)", s)
        else:
            print_dict(f"shadow vs played ({len(recs)} battles)", s["summary"])
            print_dict("referee", s["referee"])
            for title, rows in s["cases"].items():
                print(f"== {title}")
                for r in rows:
                    print("  ", json.dumps(r, ensure_ascii=False))
        return
    if args.nice:
        os.nice(args.nice)  # inherited by the Node worker
    done = load_verdicts(args.cache)
    todo = [d for d in referee_candidates(ds) if (d.battle, d.turn) not in done]
    random.Random(args.seed).shuffle(todo)
    if args.limit is not None:
        todo = todo[:args.limit]
    by_stem = dict(recs)
    sets = json.loads(Path(args.team).read_text())
    ref = Referee(sets, args.prior_bundle, args.seed, cloned=not args.no_cloned)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} positions to referee ({len(done)} cached)", flush=True)
    try:
        with open(args.cache, "a") as out:
            for i, d in enumerate(todo, 1):
                v = ref(by_stem[d.battle], d)
                out.write(json.dumps(asdict(v)) + "\n")
                out.flush()
                print(f"[{i}/{len(todo)}] {d.battle} t{d.turn} {d.kind}: {v.played} vs {v.planned} -> "
                      f"{v.verdict if v.status == 'ok' else v.status} "
                      f"{'' if v.delta is None else f'delta {v.delta:+.3f} se {v.se:.3f}'} {v.secs}s", flush=True)
    finally:
        ref.close()


if __name__ == "__main__":
    main()
