# Shadow comparison (planner vs the moves played)

`pokesim.evaluation.shadow` reads ladder records collected with `--planner shadow` and asks whether the planner's
choices would have been better than the legacy advice that actually played.

```sh
uv run python -m pokesim.evaluation.shadow summary --since 20261015-033700
uv run python -m pokesim.evaluation.shadow referee --since 20261015-033700 --until 20261015-094500
uv run python -m pokesim.evaluation.shadow report  --since 20261015-033700 --until 20261015-094500
```

`summary` uses the records only: disagreement rate (normal decisions and forced switches), the kind of each
disagreement (attack / status move / switch / Mega timing), by turn and by policy, the planner's own value gap,
and decision latency. Win rates split by how often the two disagreed are descriptive: battles with more
disagreement are different battles, not a comparison of the methods.

`referee` rebuilds each disagreeing normal decision from the protocol log (`BattleSession`, as the live
advisor) and evaluates both candidate actions with a heavier search than either method used live: 12 sampled
worlds, 2 seeds per continuation policy, 8 turns on, greedy and tactical continuations, terminal scoring without
the lost-path penalty, no deadline, the train-only prior bundle (`pokesim.data.prior_bundle`) and fresh sampler
seeds. The opponent is predicted as live (opponent model plus the cloned policy). Per position:

- `delta` = value(planner's action) − value(action played) against the prediction, with its standard error over
  worlds, and the same delta under each continuation policy alone;
- `worst_delta`: the same for each world's worst case over the opponent's options;
- verdict `planner` / `played` only when |delta| > 2 SE and all continuation policies agree on the sign.

Results are cached per (battle, turn) in `data/processed/shadow_referee.jsonl`, so the run can be stopped and
resumed. It runs at lower priority (`--nice 10`) next to a collector.

## What it does and does not show

The referee shares the planner's simulator and continuation policies. A planner win there means its choice holds
up under more worlds, seeds and longer rollouts, not that it wins more games: shared simulator errors (state
reconstruction gaps, rollout policies that play unlike people) favour both runs equally. Positions where the
continuation policies disagree are left unclear for that reason.

The records it reads are after the split manifest's `test_from`, so they are test battles. The referee fits
nothing, but its result is used to decide whether to start an A/B; the battles it reads (fixed with `--until`)
must not be reused to judge that A/B. The A/B (`--planner on` vs legacy) is judged on battles collected after it
starts.
