"""Bound deliberation by the configured allowance and the clock reported to our client."""
import re

CLOCK = re.compile(r'Time left: (\d+) sec this turn\s*\|\s*(\d+) sec total')


def decision_budget(configured_ms, history):
    budget = int(configured_ms)
    for line in reversed(history or []):
        match = CLOCK.search(line)
        if match:
            turn, total = map(int, match.groups())
            # Reserve submission time and share a dwindling total among another eight decisions.
            budget = min(budget, max(1000, (turn - 5) * 1000), max(1000, (total - 15) * 1000 // 8))
            break
    return max(100, budget)
