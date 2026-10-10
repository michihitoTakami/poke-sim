"""Name normalization and fuzzy lookup across Japanese/English names (OCR-tolerant)."""
import difflib
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import select

from pokesim.db.models import NameI18n
from pokesim.db.session import Session
from pokesim.lookup.aliases import reading


def normalize(text: str) -> str:
    """NFKC (full/half width), katakana -> hiragana, drop spaces and long-vowel variants."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    return "".join(c for c in text if not c.isspace() and c not in "-‐－・.'’:")


@dataclass(frozen=True)
class NameHit:
    kind: str
    ref_id: str
    name: str
    lang: str
    score: float


class NameIndex:
    def __init__(self, rows: list[tuple[str, str, str, str]]):
        self._exact: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        self._by_kind: dict[str, list[str]] = {}
        # Shortest id first, so a shared name ("イエッサン") resolves to the base form, not "indeedeef".
        for kind, ref_id, lang, name in sorted(rows, key=lambda r: (len(r[1]), r[1])):
            key = normalize(name)
            self._exact.setdefault((kind, key), []).append((ref_id, name, lang))
            self._exact.setdefault((kind, normalize(ref_id)), []).append((ref_id, name, lang))
            self._by_kind.setdefault(kind, []).append(key)

    @classmethod
    def from_db(cls) -> "NameIndex":
        with Session() as s:
            rows = s.execute(select(NameI18n.kind, NameI18n.ref_id, NameI18n.lang, NameI18n.name)).all()
        return cls([tuple(r) for r in rows])

    def search(self, text: str, kind: str, limit: int = 5, cutoff: float = 0.6) -> list[NameHit]:
        key = normalize(text)
        for candidate in (key, normalize(reading(text))):  # as written, then kanji -> kana reading
            if hits := self._exact.get((kind, candidate)):
                ref_id, name, lang = hits[0]
                return [NameHit(kind, ref_id, name, lang, 1.0)]
        key = normalize(reading(text))
        out: list[NameHit] = []
        seen: set[str] = set()
        for cand in difflib.get_close_matches(key, self._by_kind.get(kind, []), n=limit * 2, cutoff=cutoff):
            for ref_id, name, lang in self._exact[(kind, cand)]:
                if ref_id not in seen:
                    seen.add(ref_id)
                    out.append(NameHit(kind, ref_id, name, lang, difflib.SequenceMatcher(None, key, cand).ratio()))
        return sorted(out, key=lambda h: -h.score)[:limit]

    def resolve(self, text: str, kind: str) -> str | None:
        hits = self.search(text, kind, limit=1)
        return hits[0].ref_id if hits else None


@lru_cache(maxsize=1)
def default_index() -> NameIndex:
    return NameIndex.from_db()
