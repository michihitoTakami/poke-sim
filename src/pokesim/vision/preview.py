"""Recognise the opponent's 6 Pokémon on the Champions team preview screen.

Approach (after f77f77/pokemon-champions-assistant): fixed layout, so crop each opponent card, read its type icons
by template matching and use them (and the gender mark) to veto candidates, then rank the remaining species by
image similarity; low-confidence slots are reported as unknown rather than guessed.

- The frame is scaled to 1920x1080 (a full 16:9 game frame is assumed) and the 6 cards are found from the
  crimson card colour in the right column (falling back to the fixed grid).
- Type icons: templates cut from labelled screenshots (data/raw/vision/type_icons/<type>_<n>.png, built with
  `--build-icons`); an icon unlike every template is "unknown" and does not filter.
- Species: cosine similarity of DINOv2 (ViT-S/14) embeddings between the sprite crop and the Pokémon HOME render
  of every legal species (pokesim.vision.refs), each composited onto the card colour, plus its mirror image;
  a small usage-rate bonus (Smogon stats, megas folded into the base species) breaks near-ties.

Usage:
  uv run python -m pokesim.vision.preview data/raw/samples/preview/sample1.png [--debug out.png]
  uv run python -m pokesim.vision.preview --build-icons data/raw/samples/preview/labels.json
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from pokesim.config import RAW_DIR
from pokesim.data.smogon_stats import load_chaos, to_id
from pokesim.vision.refs import HOME_DIR, SPRITE_DIR

CANON_W, CANON_H = 1920, 1080
CARD_X0, CARD_X1 = 1559, 1854
CARD_GRID = [(155 + 126 * i, 272 + 126 * i) for i in range(6)]
# regions inside a card (x0, y0, x1, y1), card is ~295 x 117
SPRITE_BOX = (40, 2, 180, 115)
ICON_BOXES = [(188, 13, 232, 59), (238, 13, 283, 59)]  # right-aligned: a single type sits in the right slot
GENDER_BOX = (195, 75, 223, 100)
CARD_BGR = (50, 0, 105)
ICON_SIZE = 44
ICON_DIR = RAW_DIR / "vision/type_icons"
EMBED_CACHE = RAW_DIR / "vision/home_embeddings.pt"
# cosmetic formes are reported as their base species
COSMETIC = ("Vivillon", "Alcremie", "Squawkabilly", "Polteageist", "Sinistcha", "Gourgeist")
USAGE_WEIGHT, USAGE_FLOOR = 0.03, 0.002  # score += weight * log10(max(share of teams, floor))


def normalize(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.resize(img, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)


def card_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # the card is a dark crimson; bright reds (fire icons, red sprites) are not card
    return ((hsv[..., 0] >= 160) | (hsv[..., 0] <= 8)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 50) & (hsv[..., 2] < 160)


def background(img: np.ndarray) -> np.ndarray:
    """Card-coloured pixels: crimson with almost no green (orange/red sprite parts keep their green channel)."""
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    return (g < 35) & (r > 60) & (r < 170) & (b > 15) & (b < 100) & (r > b + 25)


def find_cards(img: np.ndarray) -> list[tuple[int, int]]:
    """Vertical extents of the 6 opponent cards (rows where the card's left margin is crimson)."""
    on = card_mask(img)[:, CARD_X0 + 6:CARD_X0 + 50].mean(1) > 0.6
    runs, start = [], None
    for y, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if 100 <= y - start <= 135:
                runs.append((start, y))
            start = None
    return runs if len(runs) == 6 else CARD_GRID


def sub(card: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return card[y0:y1, x0:x1]


def icon_present(icon: np.ndarray) -> bool:
    return (~background(icon)).mean() > 0.6


def gender(card: np.ndarray) -> str | None:
    g = sub(card, GENDER_BOX)
    hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
    strong = (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    blue = (strong & (hsv[..., 0] >= 100) & (hsv[..., 0] <= 130)).mean()
    red = (strong & ((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 2] > 170)).mean()
    if blue > 0.15:
        return "M"
    if red > 0.15:
        return "F"
    return None


class TypeIcons:
    def __init__(self, directory: Path = ICON_DIR):
        self.templates: list[tuple[str, np.ndarray]] = []
        for path in sorted(directory.glob("*.png")):
            self.templates.append((path.stem.rsplit("_", 1)[0], self.prep(cv2.imread(str(path)))))

    @staticmethod
    def prep(icon: np.ndarray) -> np.ndarray:
        return cv2.resize(icon, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)

    def classify(self, icon: np.ndarray, threshold: float = 0.7) -> tuple[str, float]:
        """Best-matching type, or 'unknown' when no template correlates above the threshold."""
        x = self.prep(icon)
        best, score = "unknown", -1.0
        for name, tpl in self.templates:
            s = float(cv2.matchTemplate(x, tpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            if s > score:
                best, score = name, s
        return (best if score >= threshold else "unknown"), score

    def read(self, card: np.ndarray) -> list[str]:
        icons = [sub(card, b) for b in ICON_BOXES]
        return [self.classify(icon)[0] for icon in icons if icon_present(icon)]


def square_on_background(img: np.ndarray, fg: np.ndarray, size: int = 224, pad: float = 0.08) -> np.ndarray:
    """Crop to the foreground's bounding box and centre it on a square of the card colour."""
    ys, xs = np.nonzero(fg)
    if len(xs):
        img, fg = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], fg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    side = int(max(h, w) * (1 + 2 * pad)) + 1
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = CARD_BGR
    y0, x0 = (side - h) // 2, (side - w) // 2
    region = canvas[y0:y0 + h, x0:x0 + w]
    region[fg] = img[fg]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class Slot:
    species: str | None  # None = not confident
    types: list[str]
    gender: str | None
    candidates: list[tuple[str, float]] = field(default_factory=list)


class SpeciesMatcher:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(self.device).eval()
        self.legal = {s["id"]: s for s in json.loads((SPRITE_DIR / "legal.json").read_text())}
        self.ids, self.emb = self.reference_embeddings()
        self.usage = self.usage_shares()

    def usage_shares(self) -> dict[str, float]:
        """Share of teams using each legal species (megas and other formes folded into the closest legal id)."""
        try:
            chaos = load_chaos()["data"]
        except Exception:  # noqa: BLE001 - no stats: no prior
            return {}
        out: dict[str, float] = {}
        for name, entry in chaos.items():
            sid = to_id(name)
            while sid and sid not in self.legal:
                sid = sid[:-1]
            if sid:
                out[sid] = out.get(sid, 0.0) + entry["usage"]
        return out
    def embed(self, images: list[np.ndarray]) -> torch.Tensor:
        x = torch.tensor(np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images])).permute(0, 3, 1, 2)
        x = (x.float() / 255 - self.MEAN) / self.STD
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                out.append(self.model(x[i:i + 64].to(self.device)).cpu())
        return torch.nn.functional.normalize(torch.cat(out), dim=1)

    def reference_embeddings(self) -> tuple[list[str], torch.Tensor]:
        if EMBED_CACHE.exists():
            cached = torch.load(EMBED_CACHE)
            if set(cached["ids"]) >= {p.stem for p in HOME_DIR.glob("*.png")}:
                return cached["ids"], cached["emb"]
        ids, images = [], []
        for path in sorted(HOME_DIR.glob("*.png")):
            rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                continue
            fg = rgba[..., 3] > 128
            img = square_on_background(rgba[..., :3], fg)
            for variant in (img, cv2.flip(img, 1)):
                ids.append(path.stem)
                images.append(variant)
        emb = self.embed(images)
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": ids, "emb": emb}, EMBED_CACHE)
        return ids, emb

    def allowed(self, sid: str, types: list[str], sex: str | None) -> bool:
        sp = self.legal.get(sid)
        if sp is None:
            return False
        sp_types = {t.lower() for t in sp["types"]}
        known = [t for t in types if t != "unknown"]
        if known and not set(known) <= sp_types:
            return False
        if types and "unknown" not in types and len(types) != len(sp_types):
            return False
        has_female_forme = any(s["base"] == sp["base"] and s["forme"] == "F" for s in self.legal.values())
        if sex and has_female_forme:
            return (sp["forme"] == "F") == (sex == "F")
        return True

    def display_name(self, sid: str) -> str:
        sp = self.legal[sid]
        return sp["base"] if sp["base"] in COSMETIC else sp["name"]

    def rank(self, sprite: np.ndarray, types: list[str], sex: str | None, top: int = 5) -> list[tuple[str, float]]:
        fg = ~background(sprite)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        q = self.embed([square_on_background(sprite, fg)])[0]
        sims = (self.emb @ q).numpy()
        best: dict[str, float] = {}
        for sid, s in zip(self.ids, sims):
            if self.allowed(sid, types, sex):
                name = self.display_name(sid)
                score = float(s) + USAGE_WEIGHT * np.log10(max(self.usage.get(sid, 0.0), USAGE_FLOOR))
                best[name] = max(best.get(name, -9.0), score)
        return sorted(best.items(), key=lambda kv: -kv[1])[:top]


class PreviewReader:
    def __init__(self, min_sim: float = 0.35, min_margin: float = 0.02):
        self.icons = TypeIcons()
        self.matcher = SpeciesMatcher()
        self.min_sim, self.min_margin = min_sim, min_margin

    def cards(self, frame: np.ndarray) -> list[np.ndarray]:
        img = normalize(frame)
        return [img[y0:y1, CARD_X0:CARD_X1] for y0, y1 in find_cards(img)]

    def read(self, frame: np.ndarray) -> list[Slot]:
        slots = []
        for card in self.cards(frame):
            types, sex = self.icons.read(card), gender(card)
            ranked = self.matcher.rank(sub(card, SPRITE_BOX), types, sex)
            confident = bool(ranked) and ranked[0][1] >= self.min_sim and (
                len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= self.min_margin)
            slots.append(Slot(ranked[0][0] if confident else None, types, sex, ranked))
        return slots


def build_icons(labels_path: Path) -> None:
    """Cut type-icon templates from labelled screenshots: {"file.png": [["fire"], ["bug", "water"], ...]}."""
    labels = json.loads(labels_path.read_text())
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for fname, rows in labels.items():
        img = normalize(cv2.imread(str(labels_path.parent / fname), cv2.IMREAD_UNCHANGED))
        for (y0, y1), types in zip(find_cards(img), rows):
            card = img[y0:y1, CARD_X0:CARD_X1]
            boxes = ICON_BOXES if len(types) == 2 else ICON_BOXES[1:]
            for t, box in zip(types, boxes):
                counts[t] = counts.get(t, 0) + 1
                cv2.imwrite(str(ICON_DIR / f"{t}_{counts[t]}.png"), sub(card, box))
    print(f"wrote {sum(counts.values())} icons for {len(counts)} types to {ICON_DIR}: {sorted(counts)}")


def debug_image(frame: np.ndarray, slots: list[Slot], out: str) -> None:
    img = normalize(frame)
    for (y0, y1), slot in zip(find_cards(img), slots):
        cv2.rectangle(img, (CARD_X0, y0), (CARD_X1, y1), (0, 255, 0) if slot.species else (0, 0, 255), 2)
        for x0, by0, x1, by1 in [SPRITE_BOX, *ICON_BOXES, GENDER_BOX]:
            cv2.rectangle(img, (CARD_X0 + x0, y0 + by0), (CARD_X0 + x1, y0 + by1), (255, 255, 0), 1)
        cv2.putText(img, slot.species or "?", (CARD_X0 - 330, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
    cv2.imwrite(out, img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*")
    parser.add_argument("--build-icons", help="labels JSON next to the screenshots")
    parser.add_argument("--debug", help="write an annotated copy of the (first) image here")
    args = parser.parse_args()
    if args.build_icons:
        build_icons(Path(args.build_icons))
    if not args.images:
        return
    reader = PreviewReader()
    for path in args.images:
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        slots = reader.read(frame)
        print(path)
        for i, s in enumerate(slots, 1):
            cands = ", ".join(f"{n} {v:.2f}" for n, v in s.candidates[:3])
            print(f"  {i}: {s.species or '(unknown)':20s} types={s.types} gender={s.gender}  [{cands}]")
        if args.debug and path == args.images[0]:
            debug_image(frame, slots, args.debug)


if __name__ == "__main__":
    main()
