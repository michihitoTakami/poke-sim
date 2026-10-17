"""The forme a Pokémon is in now. poke-env keeps the species id on a forme change (|detailschange|, |-formechange|)
and only swaps stats, types and ability, so Mimikyu-Busted (same stats as Mimikyu) cannot be told apart. Importing
this module patches Pokemon.forme_change to also remember the new details, read back by current_forme()."""
from functools import cache

from poke_env.battle import Pokemon
from poke_env.data import GenData

_original = Pokemon.forme_change
_original_mega = Pokemon.mega_evolve
_original_transform = Pokemon.transform


def _forme_change(self, species: str):
    _original(self, species)
    entry = GenData.from_gen(self._gen).pokedex.get("".join(c for c in species.split(",")[0].lower() if c.isalnum()))
    # Remember the exact Mega as well: a sampled held item cannot identify an observed X/Y/Z forme.
    if entry and (entry.get("battleOnly") or (entry.get("forme") or "").startswith("Mega")):
        self._last_details = species  # what poke-env compares the next |switch| details with


if getattr(Pokemon.forme_change, "__name__", "") != "_forme_change":
    Pokemon.forme_change = _forme_change


@cache
def _mega_formes(gen):
    return {"".join(c for c in entry.get('requiredItem', '').lower() if c.isalnum()): entry['name']
            for entry in GenData.from_gen(gen).pokedex.values()
            if (entry.get('forme') or '').startswith('Mega') and entry.get('requiredItem')}


def _mega_evolve(self, stone: str):
    # Upstream tries <base>mega first, even for Lucarionite Z. The later -mega event then overwrites
    # the correct stats just received in detailschange with the ordinary Mega's stats.
    forme = _mega_formes(self.gen).get(''.join(c for c in stone.lower() if c.isalnum()))
    observed = GenData.from_gen(self.gen).pokedex.get(current_forme(self), {})
    if not forme and (observed.get('forme') or '').startswith('Mega'):
        forme = observed['name']
    if forme:
        self.temporary_ability = None
        self.forme_change(forme)
    else:
        _original_mega(self, stone)


if getattr(Pokemon.mega_evolve, '__name__', '') != '_mega_evolve':
    Pokemon.mega_evolve = _mega_evolve


def current_forme(mon: Pokemon) -> str:
    """Species id of the forme it is in (e.g. "mimikyubusted"), else its species id."""
    details = mon._last_details.split(",")[0] if mon._last_details else ""
    return "".join(c for c in details.lower() if c.isalnum()) or mon.species


def _transform(self, into):
    _original_transform(self, into)
    # The target's species id can still be its base after a Mega change. Copy its CURRENT public forme.
    self._temporary_base_stats = dict(into.base_stats)
    self._temporary_types = list(into.types)


if getattr(Pokemon.transform, '__name__', '') != '_transform':
    Pokemon.transform = _transform
