# Opponent set inference (pokesim.belief)

The lookahead samples the opponent's sets from a prior (article sets, usage stats, our past battles) given the
moves, item and ability revealed. It used to ignore everything else the battle showed: a Gliscor that had used
Swords Dance and hit hard was still sampled mostly as a defensive set, so the search believed our Skeledirge would
survive the next Earthquake and chose Slack Off.

`pokesim.belief` keeps, per opponent Pokémon, a probability over candidate sets and updates it with:

- **damage taken**: our HP is exact, so the share of the simulator's 16 damage rolls that give the HP we lost (all
  rolls at least our HP for a KO);
- **damage dealt**: its HP is a percentage, so the share of our rolls within rounding of the percent it lost;
- **move order**: when both used a move of the same priority, whether the candidate's speed explains who went first;
- **no Choice item**: two different moves in one stay on the field.

Candidates are drawn from the same prior the lookahead used (48 per Pokémon, conditioned on what is revealed) and
weighted by prior x product of likelihoods. Each likelihood is floored at 0.02 so effects the reconstruction does not
carry cannot rule a set out for good. Damage and speed come from the Showdown simulator itself
(`services/sim-tools/belief.js`, the same reconstruction as the lookahead), so abilities, items, weather, screens,
burns and boosts apply. Skipped: multi-hit moves, hits on a Substitute, Focus Sash / Sturdy / Endure / Disguise
survivals, damage "[from]" an item or ability, turns with a switch before both moved, Quick Claw / Custap / Quash /
Trick Room starting, and opponent status moves (Prankster).

The lookahead draws the opponent's revealed Pokémon from the posterior when the advisor runs with `belief=True`
(`scripts/ladder.py --belief`). The advice status shows the active's most likely item / nature / spread. Off by
default.

```sh
uv run python -m pokesim.belief.evaluate --split development
uv run python scripts/ladder.py --planner on --belief ...
```

## Evaluation

Each observation is scored before it is seen, with the candidates drawn given what was revealed at that moment:
log P under equal weights (the prior the lookahead used) and under the posterior from the earlier observations.
Train-only prior bundle; battles resampled for the interval.

| records | observation | n | prior | posterior | gain (95%) |
|---|---|---|---|---|---|
| development (18 battles) | damage taken | 139 | −2.204 | −1.987 | +0.217 (+0.087, +0.361) |
| | damage dealt | 71 | −1.402 | −1.274 | +0.128 (−0.013, +0.241) |
| | move order | 116 | −0.258 | −0.220 | +0.038 (−0.001, +0.088) |
| 2026-10-15 shadow battles (151) | damage taken | 996 | −2.185 | −2.113 | +0.072 (+0.042, +0.105) |
| | damage dealt | 507 | −1.528 | −1.443 | +0.085 (+0.047, +0.126) |
| | move order | 740 | −0.274 | −0.232 | +0.042 (+0.014, +0.075) |

This measures the set inference, not playing strength. On 30 planner positions where we healed and fainted the same
turn, re-planning with the posterior changed the choice in 10, and healing went from 19 to 16; the search's own
sampling noise (6 worlds) is of the same order, so the effect on decisions needs the ladder A/B.

Known limits: the rest of a candidate's set (moves not yet seen) is still drawn from marginal frequencies; teammates
do not inform each other; mid-turn changes before a hit (our own boosts from a move earlier the same turn are
tracked, an unannounced item effect is not) can mislead a single observation, which the floor bounds.
