# Frozen offline inputs (PR 3)

Live collection may continue while these commands run. Keep the split manifest fixed; newly collected battles
after its cutoff belong to test. Do not regenerate the split just to enlarge train.

```sh
uv run python -m pokesim.data.prior_bundle
uv run python -m pokesim.rl.human build --records-only
uv run python scripts/build_observed_teams.py
uv run python scripts/eval_records.py query --split development
```

All builders accept `--ledger` and `--split-manifest`. Generated offline artifacts live under
`data/processed/scoped`; collection and live prior files are untouched. The ledger's primary record is the only
record consumed per battle. Ambiguous duplicate groups and explicitly confirmed bots are quarantined; unknown
opponents remain eligible. Unknown outcomes retain imitation decisions but mask result learning.

The prior bundle freezes published set summaries plus eligible train observations. Development, calibration and
test observations never update it. Teachers, record evaluators, imitation builders and offline training share
this bundle. Public replay imitation retains its replay-level hash holdout and excludes our ledger's Showdown
battles; real battles always use the manifest. A real development split with no decisions is an error.

Each artifact has a `.manifest.json` sidecar containing its checksum, train record hashes, split hash,
observation code hashes and prior-bundle hash where applicable. Readers reject legacy/unversioned artifacts,
changed observations, altered data and incompatible splits. Rebuild dependencies after a code change.

Example imitation fit: `uv run python -m pokesim.rl.human fit --data
data/processed/scoped/human_records.npz` (put the command on one line). To combine public replays with real
opponents, first build without `--records-only`, then fit with `--records data/processed/scoped/human_records.npz`.

This establishes data isolation, not an accuracy improvement. The public corpus and simulator database are
external inputs; their collection quality still needs auditing. Artifact checks are compatibility checks,
not a complete reproducible environment lock. Final test opening remains an explicit operational decision.
