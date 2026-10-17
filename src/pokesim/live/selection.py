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
from functools import lru_cache
from pathlib import Path

import numpy as np

from pokesim.config import PROCESSED_DIR, RAW_DIR
from pokesim.rl.myteam import to_id

PICK_RATES = PROCESSED_DIR / "pick_rates.json"
LINEUP_PRIOR = PROCESSED_DIR.parent / "lineup_prior.json"  # data/lineup_prior.json: evidence kept from lost records
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
                    not rec.get('lineup') or len(rec['lineup']) != 3 or not source or source in seen or
                    not consistent_lineup(rec)):
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
    """Offset + lead/preview interactions + the trio's mean member/preview interactions.
    The offset is the Beta posterior's logit; both coefficient tables use ridge regularisation (PREVIEW_L2).
    Unseen members or opponent species contribute no learned adjustment; static coverage can still rank them.
    """

    def __init__(self, species: list[str], leads: list[str], l2: float = PREVIEW_L2):
        self.species, self.leads, self.l2 = list(species), list(leads), l2
        self.theta = np.zeros((len(self.leads), len(self.species)))
        self.member_theta = np.zeros_like(self.theta)

    def design(self, lead: str, preview: list[str]) -> tuple[int, np.ndarray]:
        shown = set(preview)
        return (self.leads.index(lead) if lead in self.leads else -1,
                np.array([s in shown for s in self.species], dtype=float))

    def fit(self, rows: list[dict], offsets: list[float], steps: int = 300, lr: float = 0.05) -> None:
        self.theta[:] = 0
        self.member_theta[:] = 0
        designs = [(self.design(r['ordered'][0], r['preview']), o, float(r['win']), r['trio'])
                   for r, o in zip(rows, offsets)]
        designs = [(lead, x, o, y, members) for (lead, x), o, y, members in designs if lead >= 0]
        if not designs:
            return
        lead = np.array([d[0] for d in designs])
        x = np.array([d[1] for d in designs])
        offset = np.array([d[2] for d in designs])
        y = np.array([d[3] for d in designs])
        members = np.array([[float(s in d[4]) / 3 for s in self.leads] for d in designs])
        for _ in range(steps):
            logits = offset + np.einsum('ij,ij->i', self.theta[lead], x)
            logits += np.einsum('ij,jk,ik->i', members, self.member_theta, x)
            p = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            grad = self.l2 * self.theta
            np.add.at(grad, lead, (p - y)[:, None] * x)
            self.theta -= lr * grad / len(designs)
            member_grad = self.l2 * self.member_theta + members.T @ ((p - y)[:, None] * x)
            self.member_theta -= lr * member_grad / len(designs)

    def adjust(self, lead: str, preview: list[str], lineup: list[str] | None = None) -> float:
        """The fitted logit adjustment for the lead and, when given, all three brought members."""
        index, x = self.design(lead, preview)
        shift = 0.0 if index < 0 else float(self.theta[index] @ x)
        if lineup:
            members = np.array([float(s in lineup) / 3 for s in self.leads])
            shift += float(members @ self.member_theta @ x)
        return shift


@lru_cache(maxsize=1)
def _battle_formes():
    path = PROCESSED_DIR / 'dex.json'
    if not path.exists():
        return {}
    return {s['id']: to_id(s.get('battleOnly') or s.get('changesFrom') or s['name'])
            for s in json.loads(path.read_text())['species'] if isinstance(s.get('battleOnly'), (str, type(None)))}


def consistent_lineup(rec):
    """Keep unreliable roster labels out of both the empirical selection model and its preview adjustments."""
    selected = set(map(to_id, rec['lineup']))
    if len(selected) != 3 or not selected <= set(map(to_id, rec.get('team', []))):
        return False
    formes = _battle_formes()
    for line in rec.get('log', []):
        parts = line.split('|')
        if len(parts) > 3 and parts[1] in ('switch', 'drag') and parts[2].startswith('p1'):
            sid = to_id(parts[3].split(',')[0])
            if formes.get(sid, sid) not in selected:
                return False
    return True


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
                not rec.get('lineup') or len(rec['lineup']) != 3 or not source or source in seen or
                not consistent_lineup(rec)):
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
                 preview: bool = True, fit_rows: int = 200, prior: dict | None = None):
        """preview: also fit the logistic adjustment on the opponent's preview (PreviewLogistic) from the most
        recent fit_rows games, each offset by the posterior of the games before it.
        prior: earlier evidence kept as counts ({"weight": w, "ordered": [[lead,second,third wins games], ...]},
        see LINEUP_PRIOR), counted with its weight as if older than every row."""
        self.rows, self.half_life, self.kappa = rows, half_life, kappa
        weights = [0.5 ** (age / half_life) for age in range(len(rows))]
        self.games = sum(weights)
        wins = sum(w * r['win'] for w, r in zip(weights, rows))
        self.trio: dict[tuple, list[float]] = {}
        self.ordered: dict[tuple, list[float]] = {}
        for w, r in zip(weights, rows):
            for table, key in ((self.trio, r['trio']), (self.ordered, r['ordered'])):
                cell = table.setdefault(key, [0.0, 0.0])
                cell[0] += w * r['win']
                cell[1] += w
        self.prior_games = 0.0
        for key, won, games in (prior or {}).get('ordered', []):
            ids = tuple(key.split(','))
            w = float((prior or {}).get('weight', 1.0)) * 0.5 ** (len(rows) / half_life)
            ordered, trio = (ids[0], *sorted(ids[1:])), tuple(sorted(ids))
            for table, k in ((self.trio, trio), (self.ordered, ordered)):
                cell = table.setdefault(k, [0.0, 0.0])
                cell[0] += w * won
                cell[1] += w * games
            wins += w * won
            self.games += w * games
            self.prior_games += w * games
        self.global_rate = (1 + wins) / (2 + self.games)
        self.sha = hashlib.sha256(json.dumps({'version': 'lineup-members-v2', 'prior': prior,
                                             'half_life': half_life, 'kappa': kappa,
                                             'rows': [(r['name'], list(r['ordered']), r['win'], r['preview'])
                                                      for r in rows]}, sort_keys=True).encode()).hexdigest()[:16]
        self.logistic = None
        if preview and rows:
            species = [s for s, _ in Counter(s for r in rows for s in r['preview']).most_common(PREVIEW_SPECIES)]
            recent = rows[:fit_rows]
            offsets = [logit(LadderLineupModel(rows[i + 1:], half_life, kappa, preview=False)
                             .mean(list(r['ordered']))) for i, r in enumerate(recent)]
            self.logistic = PreviewLogistic(species, sorted({s for r in rows for s in r['trio']}))
            self.logistic.fit(recent, offsets)

    @classmethod
    def from_records(cls, root: Path, roster: list[str], prior_path: Path | None = None, **kw) -> "LadderLineupModel":
        """prior_path (default LINEUP_PRIOR when it exists): saved evidence counted behind the records."""
        prior_path = LINEUP_PRIOR if prior_path is None else prior_path
        prior = json.loads(prior_path.read_text()) if prior_path and prior_path.exists() else None
        return cls(ladder_rows(root, roster), prior=prior, **kw)

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
        return sigmoid(logit(a / (a + b)) + self.logistic.adjust(to_id(lineup[0]), [to_id(s) for s in preview],
                                                               list(map(to_id, lineup))))

    def evidence(self, lineup: list[str]) -> dict:
        ids = [to_id(x) for x in lineup]
        ordered, trio = (ids[0], *sorted(ids[1:])), tuple(sorted(ids))
        return {'ordered_games': round(self.ordered.get(ordered, (0.0, 0.0))[1], 2),
                'trio_games': round(self.trio.get(trio, (0.0, 0.0))[1], 2)}

    def candidates(self, roster, reference=None, picks=None, coverage=None):
        """All 60 distinct lead/trio choices; coverage is a ranking adjustment, not a calibrated win rate."""
        preview = [to_id(s) for s in picks.species] if picks is not None else None
        candidates = []
        for trio in itertools.permutations(range(len(roster)), 3):
            if list(trio[1:]) != sorted(trio[1:]):
                continue
            names = [roster[i] for i in trio]
            a, b = self.posterior(names)
            shift = self.logistic.adjust(to_id(names[0]), preview, list(map(to_id, names))) \
                if preview and self.logistic else 0.0
            coverage_info = None
            if coverage is not None and picks is not None:
                adjustment, coverage_info = coverage.score(names, picks)
                shift += adjustment
            tie = 0.0
            if reference is not None and picks is not None:
                tie = reference.win_prob(','.join(to_id(n) for n in names), picks) or 0.0
            candidates.append((names, a, b, sigmoid(logit(a / (a + b)) + shift), shift, tie, coverage_info))
        return candidates

    def rank(self, roster, picks, coverage=None, reference=None):
        return [(c[0], c[3], c[6]) for c in sorted(self.candidates(roster, reference, picks, coverage),
                                                 key=lambda c: (c[3], c[5]), reverse=True)]

    def choose(self, roster: list[str], rng, reference: "MatchupModel | None" = None,
               picks: OpponentPicks | None = None, explore: bool = True, coverage=None) -> tuple[list[str], dict]:
        """Thompson selection among competitive scores; static coverage never becomes a win-rate estimate."""
        candidates = self.candidates(roster, reference, picks, coverage)
        preview = [to_id(s) for s in picks.species] if picks is not None else None
        best = max(c[3] for c in candidates)
        pool = [c for c in candidates if c[3] >= best - LINEUP_MARGIN]
        scored = []
        for names, a, b, m, shift, tie, coverage_info in pool:
            draw = sigmoid(logit(rng.betavariate(LINEUP_TEMPER * a, LINEUP_TEMPER * b)) + shift) if explore else m
            scored.append((draw, m, tie, names, coverage_info))
        sampled, m, tie, names, coverage_info = max(scored, key=lambda s: (s[0], s[1], s[2]))
        return names, {'method': 'ladder-thompson' if explore else 'ladder-mean',
                       'posterior_mean': round(self.mean(names, preview), 4),
                       'selection_score': round(m, 4), 'score_kind': 'preview-rank-utility-v1',
                       'sampled': round(sampled, 4), 'best_mean': round(best, 4), 'pool': len(pool),
                       'preview_adjusted': self.logistic is not None and preview is not None,
                       'games': round(self.games, 2), 'records': len(self.rows),
                       'prior_games': round(self.prior_games, 2), 'sha': self.sha,
                       'half_life': self.half_life, 'kappa': self.kappa, 'temper': LINEUP_TEMPER,
                       'margin': LINEUP_MARGIN, 'coverage': coverage_info, **self.evidence(names)}


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
