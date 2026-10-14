"""The forme a Pokémon is in now. poke-env keeps the species id on a forme change (|detailschange|, |-formechange|)
and only swaps stats, types and ability, so Mimikyu-Busted (same stats as Mimikyu) cannot be told apart. Importing
this module patches Pokemon.forme_change to also remember the new details, read back by current_forme()."""
from poke_env.battle import Pokemon

_original = Pokemon.forme_change


def _forme_change(self, species: str):
    _original(self, species)
    self._last_details = species  # what poke-env compares the next |switch| details with


if getattr(Pokemon.forme_change, "__name__", "") != "_forme_change":
    Pokemon.forme_change = _forme_change


def current_forme(mon: Pokemon) -> str:
    """Species id of the forme it is in (e.g. "mimikyubusted"), else its species id."""
    details = mon._last_details.split(",")[0] if mon._last_details else ""
    return "".join(c for c in details.lower() if c.isalnum()) or mon.species
