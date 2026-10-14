# State restoration boundary (PR 4)

`services/sim-tools/state_codec.js` owns simulator reconstruction, including switch entry and Baton Pass.
`state/codec.py` exports known counters. `state/tracker.py` retains public Disable and Encore move identities
for reconstructed sessions and clears them on switching or the corresponding end event.

The active Pokémon's PP comes from its actual/generated request. Bench PP transfers consumption instead of
poke-env's maximum: standard-generation move maxima differ from Champions. Opponent exact PP is not exported.
Badly poisoned counters and known Choice locks survive reconstruction. The adapter refreshes DisableMove
events before making requests; makeRequest alone reads stale cached flags.

Synthetic fixtures may supply `pp`, `ppUsed`, `lastMove`, `choiceMove`, `disabledMove`, `disableTurns`, `encore`,
`encoreTurns`, `substituteHp`, `sleepRemaining`, `toxicStage`, and `volatileTurns`. Production observations never
invent exact substitute HP or remaining sleep time. Missing hidden counters retain simulator samples; they
are not an exact reconstruction guarantee. Timers and uncertain move locks need a richer belief tracker later.

E0 tests run an independent simulator through actual moves, extract synthetic state, restore it through the
production adapter, then compare requests and transitions using the same RNG. Scenarios include Choice,
Disable, Encore, damaged Substitute, Toxic and sleep. Existing switch-entry tests cover hazards, Intimidate,
Mega persistence and Baton transfer, plus rejection/stall handling. Invalid PP fails the branch; zero PP
produces Struggle.

This is the first E0 slice, not completion of every mechanic in the design. Hidden duration inference,
Illusion/Transform identity changes, item-transfer lock timing, source-dependent volatile effects, and all
mid-turn phases still require fixtures and tracking. These tests establish reconstruction correctness for
covered states, not opponent prediction quality or competitive strength.

```sh
uv run pytest -q tests/test_state_codec.py tests/test_worker_transitions.py tests/test_continuations.py
```

After merging, rebuild offline artifacts: session/codec changes deliberately invalidate observation manifests.
