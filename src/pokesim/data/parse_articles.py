"""Extract Pokémon sets (species / item / ability / nature / moves / stat points) from cached team articles.

Rule-based and conservative: a set starts at a line naming a species (optionally "@item"),
and the following lines are scanned for labelled fields and stat-point patterns.

Usage: uv run python -m pokesim.data.parse_articles --season M-6
"""
import argparse
import re
from dataclasses import dataclass, field

from selectolax.lexbor import LexborHTMLParser
from sqlalchemy import delete, select

from pokesim.data.fetch_articles import cache_path
from pokesim.db.models import ArticleSet, BuildArticle, Item, Learnset, Species
from pokesim.db.session import Session
from pokesim.lookup.names import NameIndex, default_index, normalize

LABELS = {
    "ability": r"特性|とくせい",
    "item": r"持ち物|もちもの|持物|道具|アイテム",
    "nature": r"性格|せいかく",
    "moves": r"技構成|技|わざ",
    "stats": r"努力値|能力ポイント|ポイント|実数値|配分|振り",
}
LABEL_RE = re.compile(rf"^[【\[［(（]?\s*(?P<label>{'|'.join(LABELS.values())})\s*[】\]］)）]?"
                      r"\s*[:：\s　]*(?P<value>.*)$")
HEADER_RE = re.compile(r"^(?P<species>[^@＠\s:：]{2,14}?)(?:\s*[（(][^)）]*[)）])?\s*[@＠]\s*(?P<item>[^\s　(（]+)")
TOKEN_SPLIT = re.compile(r"[\s　/／・,、|｜]+")
# "215(32)-132-170(17)-x-107(17)-67": actual stat with points in parentheses (no parentheses = 0 points)
_CELL = r"\d{1,3}(?:[（(]\d{1,2}[)）])?|[xXｘＸ×]"
STAT_SEQ_RE = re.compile(rf"(?:(?:{_CELL})[-－ー―]){{5}}(?:{_CELL})")
STAT_CELL_RE = re.compile(r"\d{1,3}(?:[（(](\d{1,2})[)）])?|[xXｘＸ×]")
LETTER_POINTS_RE = re.compile(r"([HABCDSＨＡＢＣＤＳ])\s*(\d{1,2})(?!\d)")
STAT_LETTERS = "HABCDS"
MAX_TOTAL_POINTS = 66
MIN_TOTAL_POINTS = 60  # players spend (almost) all 66 points; lower totals are usually false matches
MAX_STAT_POINTS = 32


@dataclass
class ParsedSet:
    species_id: str
    item_id: str | None = None
    ability_id: str | None = None
    nature: str | None = None
    moves: list[str] = field(default_factory=list)
    stat_points: list[int] | None = None
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def score(self) -> int:
        return sum(bool(x) for x in (self.item_id, self.ability_id, self.nature, self.stat_points)) + len(self.moves)


def article_lines(html: str) -> list[str]:
    tree = LexborHTMLParser(html)
    body = tree.css_first(".entry-content") or tree.body
    for img in body.css("img"):  # keep species icons that carry a usable alt text
        img.replace_with(f" {img.attributes.get('alt') or ''} ")
    text = body.text(separator="\n")
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_exact(index: NameIndex, text: str, kind: str) -> str | None:
    hits = index.search(text, kind, limit=1, cutoff=0.85)
    return hits[0].ref_id if hits and hits[0].score >= 0.85 else None


def parse_stat_points(text: str) -> list[int] | None:
    joined = re.sub(r"\s+", "", text)
    if m := STAT_SEQ_RE.search(joined):
        cells = STAT_CELL_RE.findall(m.group(0))
        points = [int(c) if c else 0 for c in cells]
        if len(points) == 6 and valid_points(points):
            return points
    for line in text.splitlines():  # "H32 A2 S32" style, all on one line
        pairs = LETTER_POINTS_RE.findall(line)
        if len(pairs) >= 2:
            points = [0] * 6
            for letter, value in pairs:
                points[STAT_LETTERS.index(normalize(letter).upper())] = int(value)
            if valid_points(points):
                return points
    return None


def valid_points(points: list[int]) -> bool:
    return MIN_TOTAL_POINTS <= sum(points) <= MAX_TOTAL_POINTS and max(points) <= MAX_STAT_POINTS


def parse_moves(index: NameIndex, text: str) -> list[str]:
    moves = []
    for token in TOKEN_SPLIT.split(text):
        if token and (move := resolve_exact(index, token, "move")) and move not in moves:
            moves.append(move)
    return moves[:4]


def species_header(index: NameIndex, line: str) -> tuple[str, str | None] | None:
    if m := HEADER_RE.match(line):
        if species := resolve_exact(index, m.group("species"), "species"):
            return species, resolve_exact(index, m.group("item"), "item")
    stripped = re.sub(r"[（(][^)）]*[)）]|[【】\[\]［］■◆●▼★☆・\d.．①-⑳]", "", line).strip()
    if 2 <= len(stripped) <= 12:  # a bare name line must match exactly (fuzzy hits catch abilities like 鮫肌)
        hits = index.search(stripped, "species", limit=1, cutoff=0.85)
        if hits and hits[0].score == 1.0:
            return hits[0].ref_id, None
    return None


def fill_fields(index: NameIndex, current: ParsedSet) -> None:
    pending_label = None
    for line in current.lines[1:]:
        m = LABEL_RE.match(line)
        label = next((k for k, pat in LABELS.items() if m and re.fullmatch(pat, m.group("label"))), None)
        value = m.group("value") if m else line
        if label is None and pending_label:  # value on the line after its label
            label, pending_label = pending_label, None
        elif label and not value.strip():
            pending_label = label
            continue
        if label == "ability" and not current.ability_id:
            first = re.split(r"[→>＞\s　]", value.strip())
            current.ability_id = next((a for t in reversed(first) if t and (a := resolve_exact(index, t, "ability"))),
                                      None)
        elif label == "item" and not current.item_id:
            current.item_id = resolve_exact(index, value.split()[0] if value.split() else value, "item")
        elif label == "nature" and not current.nature:
            current.nature = next((n for t in TOKEN_SPLIT.split(value) if t and
                                   (n := resolve_exact(index, t, "nature"))), None)
        elif label == "moves" and len(current.moves) < 4:
            current.moves += [mv for mv in parse_moves(index, value) if mv not in current.moves]
        elif label is None and len(current.moves) < 4:
            tokens = [t for t in TOKEN_SPLIT.split(line) if t]
            found = parse_moves(index, line)
            if len(tokens) >= 2 and len(found) == len(tokens):  # a bare line of moves
                current.moves += [mv for mv in found if mv not in current.moves]
        if not current.nature:  # natures are often written inline ("臆病C252", "性格：控えめ")
            for token in re.findall(r"[ぁ-んァ-ヶ一-龥]{2,5}", line):
                if nature := resolve_exact(index, token, "nature"):
                    current.nature = nature
                    break
    current.moves = current.moves[:4]
    current.stat_points = parse_stat_points("\n".join(current.lines[1:]))


def parse_article(index: NameIndex, html: str) -> list[ParsedSet]:
    blocks: list[ParsedSet] = []
    for line in article_lines(html):
        if header := species_header(index, line):
            blocks.append(ParsedSet(species_id=header[0], item_id=header[1], lines=[line]))
        elif blocks:
            blocks[-1].lines.append(line)
    for block in blocks:
        block.lines = block.lines[:40]  # a set description rarely runs longer; avoids eating the next section
        fill_fields(index, block)
    best: dict[str, ParsedSet] = {}  # keep the most informative block per species (skips table-of-contents hits)
    for block in blocks:
        if block.score() > 0 and (block.species_id not in best or block.score() > best[block.species_id].score()):
            best[block.species_id] = block
    return sorted(best.values(), key=blocks.index)[:6]


def normalize_species(s, parsed: ParsedSet) -> None:
    """Megas are recorded as base species + mega stone; pick the forme that has the stated ability."""
    species = s.get(Species, parsed.species_id)
    if species.is_mega:
        base = s.scalars(select(Species).where(Species.name == species.changes_from)).first()
        if base:
            if not parsed.item_id and species.required_item:
                parsed.item_id = s.scalars(select(Item.id).where(Item.name == species.required_item)).first()
            parsed.species_id, species = base.id, base
    if parsed.ability_id and parsed.ability_id not in species.abilities:
        formes = s.scalars(select(Species).where(Species.base_species == species.base_species,
                                                 Species.is_mega.is_(False))).all()
        if match := next((f for f in formes if parsed.ability_id in f.abilities), None):
            parsed.species_id = match.id


def validate(s, parsed: ParsedSet) -> list[str]:
    problems = []
    species = s.get(Species, parsed.species_id)
    if parsed.item_id and (item := s.get(Item, parsed.item_id)) and item.mega_stone:
        mega = s.get(Species, item.mega_stone)
        if mega and mega.changes_from != species.name:
            problems.append(f"mega stone {parsed.item_id} does not match {parsed.species_id}")
    legal_moves = set(s.scalars(select(Learnset.move_id).where(Learnset.species_id == parsed.species_id)))
    problems += [f"cannot learn {mv}" for mv in parsed.moves if mv not in legal_moves]
    if parsed.ability_id:
        megas = s.scalars(select(Species).where(Species.changes_from == species.name)).all()
        allowed = set(species.abilities) | {a for m in megas for a in m.abilities}
        if parsed.ability_id not in allowed:
            problems.append(f"ability {parsed.ability_id} not available")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="M-6")
    args = parser.parse_args()
    index = default_index()

    with Session.begin() as s:
        articles = s.execute(select(BuildArticle.id, BuildArticle.rank).where(BuildArticle.season == args.season)
                             .order_by(BuildArticle.rank)).all()
        cached = [(aid, rank) for aid, rank in articles if cache_path(aid).exists()]
        s.execute(delete(ArticleSet).where(ArticleSet.article_id.in_([aid for aid, _ in cached])))
        total = with_problems = 0
        for aid, rank in cached:
            sets = parse_article(index, cache_path(aid).read_text(encoding="utf-8"))
            kept = []
            for ps in sets:
                normalize_species(s, ps)
                ps.problems = validate(s, ps)
                illegal = sum(p.startswith("cannot learn") for p in ps.problems)
                if ps.moves and illegal * 2 >= len(ps.moves):  # moves belong to another (image-only) Pokémon
                    continue
                kept.append(ps)
            sets = kept
            for slot, ps in enumerate(sets):
                problems = ps.problems
                with_problems += bool(problems)
                s.add(ArticleSet(article_id=aid, slot=slot, species_id=ps.species_id, item_id=ps.item_id,
                                 ability_id=ps.ability_id, nature=ps.nature, moves=ps.moves,
                                 stat_points=ps.stat_points, problems=problems, raw_text="\n".join(ps.lines)))
            total += len(sets)
            print(f"rank {rank:>4}: {len(sets)} sets  " + " ".join(p.species_id for p in sets))
    print(f"{total} sets from {len(cached)} articles ({with_problems} with legality problems)")


if __name__ == "__main__":
    main()
