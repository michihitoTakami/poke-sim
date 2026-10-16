"""Bounded search utility from a public position and an uncalibrated reference value.

This is a move-comparison score, never a calibrated probability of beating a human opponent.
The frozen V_ref predicts play by a particular reference policy; material anchors it when that policy is optimistic.
"""
import math

from pokesim.value.features import health as observed_health

VERSION = 'reference-material-v1'
REFERENCE_WEIGHT = .5
OPEN_LIMIT = .85


def health(condition):
    hp, status, _, _ = observed_health(condition or '0 fnt')
    return hp, status


def position_value(observation: dict) -> float:
    """Use only the acting player's request and public observations, with unseen opponents explicitly unknown."""
    side = observation['side']
    foe = 'p2' if side == 'p1' else 'p1'
    own = observation['request']['side']['pokemon']
    public = observation['public']['mons']
    theirs = list(public[foe].values())
    size = len(own)
    unseen = max(0, size - len(theirs))

    def totals(mons):
        values = [health(m.get('condition')) for m in mons]
        return (sum(h > 0 for h, _ in values), sum(h for h, _ in values),
                sum(h > 0 and status in {'brn', 'par', 'psn', 'tox', 'slp', 'frz'} for h, status in values))

    alive, hp, status = totals(own)
    other_alive, other_hp, other_status = totals(theirs)
    # A never-seen selected opponent is not assumed fainted or weakened.
    other_alive += unseen
    other_hp += unseen

    def boosts(team):
        mon = public[team].get(observation['public']['active'][team], {})
        b = mon.get('boosts', {})
        # Diminishing gains, with no value assigned to stacking beyond the legal cap.
        return sum(weight * math.tanh(max(-6, min(6, b.get(stat, 0))) / 3)
                   for stat, weight in {'atk': .10, 'spa': .10, 'def': .07, 'spd': .07, 'spe': .08}.items())

    advantage = .70 * (alive - other_alive) + .35 * (hp - other_hp) + .10 * (other_status - status)
    advantage += boosts(side) - boosts(foe)
    return math.tanh(advantage)


def leaf_value(observation: dict, reference_probability: float) -> float:
    """Terminal results are handled by the simulator; open leaves cannot claim a certain win or loss."""
    if not math.isfinite(reference_probability) or not 0 <= reference_probability <= 1:
        raise ValueError('invalid reference value')
    value = REFERENCE_WEIGHT * (2 * reference_probability - 1)
    value += (1 - REFERENCE_WEIGHT) * position_value(observation)
    return max(-OPEN_LIMIT, min(OPEN_LIMIT, value))
