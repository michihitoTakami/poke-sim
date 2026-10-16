"""Which versions made a decision: code commit, policy files and the lineup table, saved with each battle record so
results can be grouped by what actually played (and a lineup table fitted for another policy is noticed).

A full battle package (policy, priors, selection model, calibration, search settings under one id, switched
atomically) is a later step; this records the pieces we already have.
"""
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

from pokesim.config import ROOT, SHOWDOWN_DIR


@lru_cache(maxsize=64)
def _file_hash(path: str, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def file_hash(path: str | Path | None) -> str | None:
    """First 16 hex digits of the file's SHA-256 (None when there is no such file)."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _file_hash(str(p.resolve()), p.stat().st_mtime)


@lru_cache(maxsize=4)
def git_commit(repo: str = str(ROOT)) -> dict:
    """{"commit": short sha, "dirty": uncommitted changes to tracked files} (empty when not a git checkout)."""
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short=12", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit, "dirty": dirty}


def describe(policy: str | Path | None = None, matchup_meta: dict | None = None, **extra) -> dict:
    """Version info for a battle record."""
    out = {"code": git_commit(), "engine": git_commit(str(SHOWDOWN_DIR))}
    if policy:
        out["policy"] = {"path": Path(policy).name, "sha": file_hash(policy)}
    if matchup_meta is not None:
        out["matchup_model"] = {k: matchup_meta.get(k) for k in ("policy_sha", "policies", "data_sha", "games",
                                                                 "fitted", "ladder_evidence")}
    out.update(extra)
    return out
