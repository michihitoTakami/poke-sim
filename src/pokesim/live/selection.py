"""Team preview: which 3 the opponent is likely to bring (and lead with), and which 3 we should bring.

Opponent: per species, how often it is brought / leads when shown, from rated Showdown replays (6 shown -> the
Pokémon that appeared). A game can end before the third member appears, so the rates are relative; for a given 6
they are scaled so the expected number brought is 3 and lead probabilities sum to 1.

Us: `MatchupModel` (fitted offline from simulated games by scripts/sim_matchups.py) gives the win chance of each of
our lineups (3 of 6, lead first) against each opponent member being brought; at preview time it is averaged over
the opponent's predicted picks.

Usage: uv run python -m pokesim.live.selection pick-rates   # rebuild the replay statistics
"""
import hashlib
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokesim.config import PROCESSED_DIR, RAW_DIR
from pokesim.rl.myteam import to_id

PICK_RATES = PROCESSED_DIR / "pick_rates.json"
MATCHUP_MODEL = PROCESSED_DIR / "matchup_model.json"
PRIOR = 2.0  # pseudo-observations at the average rate, for rarely seen species
LADDER_PRIOR = 12.0  # shrink live results towards the simulated matchup estimate
LADDER_WINDOW = 200


def build_pick_rates() -> dict:
    from sqlalchemy import select

    from pokesim.data.parse_replays import base_species, parse_log
    from pokesim.db.models import Species
    from pokesim.db.session import Session

    with Session() as s:
        species = {sp.id: sp for sp in s.scalars(select(Species))}
    shown, brought, led = Counter(), Counter(), Counter()
    for path in sorted((RAW_DIR / "replays").glob("*.json")):
        log = json.loads(path.read_text())["log"]
        sides = parse_log(log, species)
        for role, side in sides.items():
            if len(side.preview) != 6:
                continue
            first = next((line.split("|")[3] for line in log.split("\n")
                          if line.startswith(f"|switch|{role}a:")), None)
            lead = base_species(first, species) if first else None
            for sid in side.preview:
                shown[sid] += 1
                brought[sid] += sid in side.nick.values()
            if lead:
                led[lead] += 1
    total_shown = sum(shown.values())
    mean_brought = sum(brought.values()) / total_shown
    mean_led = sum(led.values()) / total_shown
    rates = {sid: {"shown": n,
                   "brought": (brought[sid] + PRIOR * mean_brought) / (n + PRIOR),
                   "lead": (led[sid] + PRIOR * mean_led) / (n + PRIOR)} for sid, n in shown.items()}
    out = {"mean": {"brought": mean_brought, "lead": mean_led}, "species": rates}
    PICK_RATES.write_text(json.dumps(out, indent=0))
    return out


def scale_to(rates: np.ndarray, total: float) -> np.ndarray:
    """Scale rates so they sum to `total` with each at most 1 (iteratively capping)."""
    p = np.asarray(rates, dtype=float).copy()
    for _ in range(20):
        free = p < 1.0
        rest = total - (~free).sum()
        if free.sum() == 0 or p[free].sum() == 0:
            break
        p[free] = p[free] * rest / p[free].sum()
        if (p <= 1.0 + 1e-9).all():
            break
        p = np.minimum(p, 1.0)
    return np.minimum(p, 1.0)


@dataclass
class OpponentPicks:
    species: list[str]
    brought: np.ndarray  # P(brought), sums to 3
    lead: np.ndarray  # P(lead), sums to 1


class PickModel:
    def __init__(self, path: Path = PICK_RATES):
        data = json.loads(path.read_text()) if path.exists() else {"mean": {"brought": 0.5, "lead": 1 / 6},
                                                                      "species": {}}
        self.mean, self.rates = data["mean"], data["species"]

    def predict(self, species: list[str]) -> OpponentPicks:
        ids = [to_id(s) for s in species]
        b = np.array([self.rates.get(s, self.mean)["brought"] for s in ids])
        lead = np.array([self.rates.get(s, self.mean)["lead"] for s in ids])
        return OpponentPicks(list(species), scale_to(b, 3.0), lead / lead.sum())


class MatchupModel:
    """logit P(win) = bias[lineup] + sum over brought opponent members of effect[lineup][species] (fitted offline).

    A lineup key is "lead,second,third" (our species ids). Unknown species fall back to the lineup's mean effect.
    `meta` says what the table was fitted from (scripts/sim_matchups.py fit): the policy that played the simulated
    games (policy_sha), the data file and when. A table fitted for another policy is a reference only."""

    def __init__(self, path: Path = MATCHUP_MODEL):
        data = json.loads(path.read_text()) if path.exists() else {"lineups": {}}
        self.lineups: dict[str, dict] = data["lineups"]
        self.meta: dict = data.get("meta", {})
        self.evidence: dict[tuple[str, ...], tuple[int, int]] = {}

    def observe_ladder(self, root: Path, roster: list[str]) -> None:
        """Adapt preview scores to recent rated games of this roster. Never mix in local bot games.

        Pool by trio because the small ladder sample cannot identify lead and matchup effects independently.
        This is a shrunk selection estimate, not a calibrated win probability or a causal lineup comparison.
        Called before preview, so only already completed games contribute.
        """
        rows = []
        seen = set()
        for path in sorted(root.glob('*.json'), reverse=True):
            try:
                rec = json.loads(path.read_text())
            except (OSError, ValueError):
                continue  # A concurrent writer or damaged record must not prevent team preview.
            if not isinstance(rec, dict):
                continue
            source = rec.get('source')
            if (rec.get('result') not in ('win', 'loss') or '|rated|' not in rec.get('log', []) or
                    sorted(map(to_id, rec.get('team', []))) != sorted(map(to_id, roster)) or
                    not rec.get('lineup') or len(rec['lineup']) != 3 or not source or source in seen):
                continue
            seen.add(source)
            rows.append((path.name, rec))
            if len(rows) == LADDER_WINDOW:
                break
        counts = {}
        for _, rec in rows:
            key = tuple(sorted(map(to_id, rec['lineup'])))
            wins, games = counts.get(key, (0, 0))
            counts[key] = wins + (rec['result'] == 'win'), games + 1
        self.evidence = counts
        self.meta['ladder_evidence'] = {
            'games': len(rows), 'window': LADDER_WINDOW, 'prior_games': LADDER_PRIOR,
            'sha': hashlib.sha256(json.dumps([(name, r['lineup'], r['result']) for name, r in rows]).encode())
                .hexdigest()[:16],
        }

    def check(self, policy: str | Path | None) -> str | None:
        """Why the table may not fit the policy that will play (None when it does, or there is no policy)."""
        from pokesim.runtime.bundle import file_hash

        if not policy or not self.lineups:
            return None
        sha = file_hash(policy)
        fitted = self.meta.get("policy_sha")
        if fitted is None:
            return "fitted before policy versions were recorded: its win chances are for an unknown policy"
        if fitted != sha:
            return f"fitted for policy {fitted}, playing {sha}: its win chances are a reference only"
        return None

    def win_prob(self, lineup: str, picks: OpponentPicks) -> float | None:
        m = self.lineups.get(lineup)
        if m is None:
            return None
        effects = m["effects"]
        default = float(np.mean(list(effects.values()))) if effects else 0.0
        logit = m["bias"] + sum(p * effects.get(to_id(s), default) for s, p in zip(picks.species, picks.brought))
        return float(1 / (1 + np.exp(-logit)))

    def rank(self, roster: list[str], picks: OpponentPicks) -> list[tuple[list[str], float]]:
        out = []
        for trio in itertools.permutations(range(len(roster)), 3):
            if list(trio[1:]) != sorted(trio[1:]):  # order of the back two does not matter
                continue
            names = [roster[i] for i in trio]
            p = self.win_prob(",".join(to_id(n) for n in names), picks)
            if p is not None:
                wins, games = self.evidence.get(tuple(sorted(map(to_id, names))), (0, 0))
                p = (LADDER_PRIOR * p + wins) / (LADDER_PRIOR + games)
                out.append((names, p))
        return sorted(out, key=lambda kv: -kv[1])


LINEUP_HALF_LIFE = 150.0  # games: ladder evidence loses half its weight after this many later games
LINEUP_KAPPA = 6.0  # pseudo-games tying an ordered lineup to its trio, and a trio to the roster's overall rate
LINEUP_MARGIN = 0.05  # Thompson sampling only among lineups whose posterior mean is within this of the best
LINEUP_TEMPER = 2.0  # Thompson draws from Beta(temper * a, temper * b): less exploration than the raw posterior
PREVIEW_SPECIES = 30  # most frequent opponent preview species the logistic adjustment conditions on
PREVIEW_L2 = 3.0


def logit(p: float) -> float:
    p = min(1 - 1e-3, max(1e-3, p))
    return float(np.log(p / (1 - p)))


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


class PreviewLogistic:
    """logit P(win) = offset + sum over the opponent's shown species of theta[our lead][species].
    The offset is the Beta posterior's logit; theta is ridge-regularised (PREVIEW_L2) and fitted by gradient
    descent on the rows given. Leads or species outside the fitted tables contribute nothing."""

    def __init__(self, species: list[str], leads: list[str], l2: float = PREVIEW_L2):
        self.species, self.leads, self.l2 = list(species), list(leads), l2
        self.theta = np.zeros((len(self.leads), len(self.species)))

    def design(self, lead: str, preview: list[str]) -> tuple[int, np.ndarray]:
        shown = set(preview)
        return (self.leads.index(lead) if lead in self.leads else -1,
                np.array([s in shown for s in self.species], dtype=float))

    def fit(self, rows: list[dict], offsets: list[float], steps: int = 300, lr: float = 0.05) -> None:
        self.theta[:] = 0
        designs = [(self.design(r['ordered'][0], r['preview']), o, float(r['win'])) for r, o in zip(rows, offsets)]
        designs = [(lead, x, o, y) for (lead, x), o, y in designs if lead >= 0]
        if not designs:
            return
        lead = np.array([d[0] for d in designs])
        x = np.array([d[1] for d in designs])
        offset = np.array([d[2] for d in designs])
        y = np.array([d[3] for d in designs])
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(offset + np.einsum('ij,ij->i', self.theta[lead], x))))
            grad = self.l2 * self.theta
            np.add.at(grad, lead, (p - y)[:, None] * x)
            self.theta -= lr * grad / len(designs)

    def adjust(self, lead: str, preview: list[str]) -> float:
        """The change of the win logit for our lead against this preview (0 without a fitted lead)."""
        index, x = self.design(lead, preview)
        return 0.0 if index < 0 else float(self.theta[index] @ x)


def ladder_rows(root: Path, roster: list[str]) -> list[dict]:
    """Completed rated ladder games of this roster, most recent first: ordered lineup, result, opponent preview.
    Local bot games, unrated games, other rosters and duplicate records of one battle are left out."""
    rows, seen = [], set()
    for path in sorted(root.glob('*.json'), reverse=True):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # A concurrent writer or damaged record must not prevent team preview.
        if not isinstance(rec, dict):
            continue
        source = rec.get('source')
        if (rec.get('result') not in ('win', 'loss') or '|rated|' not in rec.get('log', []) or
                sorted(map(to_id, rec.get('team', []))) != sorted(map(to_id, roster)) or
                not rec.get('lineup') or len(rec['lineup']) != 3 or not source or source in seen):
            continue
        seen.add(source)
        lineup = [to_id(x) for x in rec['lineup']]
        rows.append({'name': path.name, 'started': rec.get('started', ''), 'win': rec['result'] == 'win',
                     'ordered': (lineup[0], *sorted(lineup[1:])), 'trio': tuple(sorted(lineup)),
                     'preview': [to_id(s) for s in (rec.get('preview') or {}).get('species', [])]})
    return rows


class LadderLineupModel:
    """Which 3 to bring, learned from this roster's own rated ladder results.

    Hierarchical Beta posterior: the roster's overall win rate -> each trio -> each ordered lineup (lead first),
    each level a Beta prior of LINEUP_KAPPA pseudo-games at the level above. Evidence decays by half every
    LINEUP_HALF_LIFE later games, because the player that produced it keeps changing. This is a selection
    estimate from lineups chosen with the opponents in view, not a causal comparison. Lineups without any
    evidence rank by the posterior of their trio or the roster, with the simulated matchup table (when given) only
    breaking ties."""

    def __init__(self, rows: list[dict], half_life: float = LINEUP_HALF_LIFE, kappa: float = LINEUP_KAPPA,
                 preview: bool = True, fit_rows: int = 200):
        """preview: also fit the logistic adjustment on the opponent's preview (PreviewLogistic) from the most
        recent fit_rows games, each offset by the posterior of the games before it."""
        self.rows, self.half_life, self.kappa = rows, half_life, kappa
        weights = [0.5 ** (age / half_life) for age in range(len(rows))]
        self.games = sum(weights)
        wins = sum(w * r['win'] for w, r in zip(weights, rows))
        self.global_rate = (1 + wins) / (2 + self.games)
        self.trio: dict[tuple, list[float]] = {}
        self.ordered: dict[tuple, list[float]] = {}
        for w, r in zip(weights, rows):
            for table, key in ((self.trio, r['trio']), (self.ordered, r['ordered'])):
                cell = table.setdefault(key, [0.0, 0.0])
                cell[0] += w * r['win']
                cell[1] += w
        self.sha = hashlib.sha256(json.dumps([(r['name'], list(r['ordered']), r['win']) for r in rows])
                                  .encode()).hexdigest()[:16]
        self.logistic = None
        if preview and rows:
            species = [s for s, _ in Counter(s for r in rows for s in r['preview']).most_common(PREVIEW_SPECIES)]
            recent = rows[:fit_rows]
            offsets = [logit(LadderLineupModel(rows[i + 1:], half_life, kappa, preview=False)
                             .mean(list(r['ordered']))) for i, r in enumerate(recent)]
            self.logistic = PreviewLogistic(species, sorted({r['ordered'][0] for r in rows}))
            self.logistic.fit(recent, offsets)

    @classmethod
    def from_records(cls, root: Path, roster: list[str], **kw) -> "LadderLineupModel":
        return cls(ladder_rows(root, roster), **kw)

    def posterior(self, lineup: list[str]) -> tuple[float, float]:
        """Beta(alpha, beta) of winning with this ordered lineup (lead first)."""
        ids = [to_id(x) for x in lineup]
        ordered, trio = (ids[0], *sorted(ids[1:])), tuple(sorted(ids))
        wins, games = self.trio.get(trio, (0.0, 0.0))
        trio_rate = (self.kappa * self.global_rate + wins) / (self.kappa + games)
        wins, games = self.ordered.get(ordered, (0.0, 0.0))
        return self.kappa * trio_rate + wins, self.kappa * (1 - trio_rate) + games - wins

    def mean(self, lineup: list[str], preview: list[str] | None = None) -> float:
        """Posterior mean win chance; with the opponent's preview, adjusted by the logistic model."""
        a, b = self.posterior(lineup)
        if preview is None or self.logistic is None:
            return a / (a + b)
        return sigmoid(logit(a / (a + b)) + self.logistic.adjust(to_id(lineup[0]), [to_id(s) for s in preview]))

    def evidence(self, lineup: list[str]) -> dict:
        ids = [to_id(x) for x in lineup]
        ordered, trio = (ids[0], *sorted(ids[1:])), tuple(sorted(ids))
        return {'ordered_games': round(self.ordered.get(ordered, (0.0, 0.0))[1], 2),
                'trio_games': round(self.trio.get(trio, (0.0, 0.0))[1], 2)}

    def choose(self, roster: list[str], rng, reference: "MatchupModel | None" = None,
               picks: OpponentPicks | None = None, explore: bool = True) -> tuple[list[str], dict]:
        """The lineup to bring and how it was chosen. picks: the opponent's preview (its species condition the
        logistic adjustment). explore: Thompson sampling (tempered by LINEUP_TEMPER) among the lineups whose
        adjusted posterior mean is within LINEUP_MARGIN of the best; otherwise the best mean. reference: the
        simulated matchup table's win chance, used only to break ties between lineups of equal mean."""
        preview = [to_id(s) for s in picks.species] if picks is not None else None
        candidates = []
        for trio in itertools.permutations(range(len(roster)), 3):
            if list(trio[1:]) != sorted(trio[1:]):
                continue
            names = [roster[i] for i in trio]
            a, b = self.posterior(names)
            shift = self.logistic.adjust(to_id(names[0]), preview) if preview and self.logistic else 0.0
            tie = 0.0
            if reference is not None and picks is not None:
                tie = reference.win_prob(','.join(to_id(n) for n in names), picks) or 0.0
            candidates.append((names, a, b, sigmoid(logit(a / (a + b)) + shift), shift, tie))
        best = max(c[3] for c in candidates)
        pool = [c for c in candidates if c[3] >= best - LINEUP_MARGIN]
        scored = []
        for names, a, b, m, shift, tie in pool:
            draw = sigmoid(logit(rng.betavariate(LINEUP_TEMPER * a, LINEUP_TEMPER * b)) + shift) if explore else m
            scored.append((draw, m, tie, names))
        sampled, m, tie, names = max(scored, key=lambda s: (s[0], s[1], s[2]))
        return names, {'method': 'ladder-thompson' if explore else 'ladder-mean', 'posterior_mean': round(m, 4),
                       'sampled': round(sampled, 4), 'best_mean': round(best, 4), 'pool': len(pool),
                       'preview_adjusted': self.logistic is not None and preview is not None,
                       'games': round(self.games, 2), 'records': len(self.rows), 'sha': self.sha,
                       'half_life': self.half_life, 'kappa': self.kappa, 'temper': LINEUP_TEMPER,
                       'margin': LINEUP_MARGIN, **self.evidence(names)}


def main() -> None:
    if sys.argv[1:] == ["pick-rates"]:
        out = build_pick_rates()
        top = sorted(out["species"].items(), key=lambda kv: -kv[1]["shown"])[:15]
        mean = out["mean"]
        print(f"{len(out['species'])} species; mean brought {mean['brought']:.2f}, lead {mean['lead']:.2f}")
        for sid, r in top:
            print(f"  {sid:18s} shown {r['shown']:4d}  brought {r['brought']:.2f}  lead {r['lead']:.2f}")


if __name__ == "__main__":
    main()
