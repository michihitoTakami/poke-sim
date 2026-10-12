"""Reading the battle screen (1920x1080 capture): the message box, the two HP panels and the command prompt.

Coordinates were measured on captures of Pokémon Champions (Switch 2, 1080p capture card):
  message box   white text on a dark band, bottom-left; a small furigana row sits above kanji
  opponent HUD  top-right: name plate, HP bar, "NN%"
  our HUD       bottom-left: name plate, HP bar, "cur/max"
  COMMAND       the turn timer label at the right edge; on screen whenever we have to choose
                (main menu, move list, party/switch screen, status screens)

OCR: EasyOCR (Japanese) for text, manga-ocr for the italic HP digits. Both load lazily on the GPU.
`norm` folds the usual OCR confusions (small kana, ー/一, 力/カ, 三/ミ ...) so text can be matched against names.
"""
import re
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

from pokesim.vision.preview import normalize

MSG_BOX = (250, 770, 1500, 930)
OPP_NAME = (1590, 50, 1830, 95)
OPP_PCT = (1700, 115, 1845, 170)
OPP_BAR = (117, 1621, 1844)  # row, x at 0%, x at 100%
MY_NAME = (150, 930, 390, 980)
MY_HP = (240, 1000, 400, 1055)
MY_BAR = (1000, 166, 403)
MY_BAR_OFFSET = 0.022  # the bar's slanted end makes the fill look ~2% longer than the HP (measured against the digits)
COMMAND = (1580, 100, 1760, 350)

SMALL = str.maketrans("ァィゥェォッャュョヮヵヶぁぃぅぇぉっゃゅょゎ", "アイウエオツヤユヨワカケあいうえおつやゆよわ")
CONFUSE = str.maketrans({"一": "ー", "－": "ー", "-": "ー", "—": "ー", "―": "ー", "~": "ー", "〜": "ー", "力": "カ",
                         "口": "ロ", "二": "ニ", "三": "ミ", "工": "エ", "夕": "タ", "卜": "ト", "八": "ハ",
                         "へ": "ヘ", "べ": "ベ", "ぺ": "ペ", "ガ": "が"})
JUNK = re.compile(r"[\s!！?？」「『』【】〔〕()（）\[\]{}。、.,・:;'\"`|_@%＠]")


def norm(text: str) -> str:
    return JUNK.sub("", unicodedata.normalize("NFKC", text)).translate(SMALL).translate(CONFUSE)


@dataclass
class Hud:
    name: str          # OCR text of the name plate
    frac: float        # HP fraction from the bar
    value: str = ""    # OCR text of "NN%" (opponent) or "cur/max" (us)


@dataclass
class Reading:
    message: str | None = None
    opp: Hud | None = None
    me: Hud | None = None
    decision: bool = False


def crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def bar_fraction(hsv: np.ndarray, bar: tuple[int, int, int]) -> float | None:
    """Fill of an HP bar: rightmost saturated pixel on the bar row. None when there is no bar (HUD hidden)."""
    y, xa, xb = bar
    row = hsv[y - 1:y + 2, xa - 3:xb + 3].mean(0)
    filled = (row[:, 1] > 90) & (row[:, 2] > 90)
    dark = (row[:, 2] < 90).mean()
    xs = np.where(filled)[0]
    if len(xs) < 3:
        return 0.0 if dark > 0.8 else None  # an empty bar is a dark track
    return min(1.0, (xs.max() - 3) / (xb - xa))


class ScreenReader:
    def __init__(self, gpu: bool = True):
        import easyocr

        self.ocr = easyocr.Reader(["ja"], gpu=gpu, verbose=False)
        self.ocr_en = easyocr.Reader(["en"], gpu=gpu, verbose=False)  # the ja model misreads Latin ("COMMAND")
        self._digits = None
        self.gpu = gpu

    @property
    def digits(self):
        if self._digits is None:
            from manga_ocr import MangaOcr

            self._digits = MangaOcr(force_cpu=not self.gpu)
        return self._digits

    def text(self, img: np.ndarray) -> str:
        return "".join(self.ocr.readtext(img, detail=0))

    def digits_text(self, img: np.ndarray) -> str:
        from PIL import Image

        return unicodedata.normalize("NFKC", self.digits(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))

    def message(self, img: np.ndarray) -> str | None:
        """The message box's text, furigana dropped (boxes much shorter than the main lines), lines joined."""
        reg = crop(img, MSG_BOX)
        hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 1] < 50) & (hsv[..., 2] > 200)).mean()
        if not 0.01 < white < 0.3:
            return None
        boxes = self.ocr.readtext(reg, detail=1, paragraph=False)
        if not boxes:
            return None
        rows = []
        for pts, text, _conf in boxes:
            ys = [p[1] for p in pts]
            xs = [p[0] for p in pts]
            rows.append(((min(ys) + max(ys)) / 2, min(xs), max(ys) - min(ys), text))
        height = max(r[2] for r in rows)
        rows = [r for r in rows if r[2] >= 0.65 * height and not re.fullmatch(r"\W*\d{1,2}[:.]\d{2}\W*", r[3])]
        lines: list[list[tuple]] = []
        for r in sorted(rows):  # by vertical centre; boxes within half a line height share a line
            if lines and abs(r[0] - lines[-1][0][0]) < 0.5 * height:
                lines[-1].append(r)
            else:
                lines.append([r])
        text = " ".join(r[3] for line in lines for r in sorted(line, key=lambda r: r[1])).strip()
        return text or None

    def hud(self, img: np.ndarray, hsv: np.ndarray, name_box, bar, value_box, offset: float = 0.0) -> Hud | None:
        frac = bar_fraction(hsv, bar)
        if frac is None:
            return None
        name = self.text(crop(img, name_box))
        if not name:
            return None
        if offset and frac < 0.995:
            frac = max(0.0, frac - offset)
        return Hud(name=name, frac=frac, value=self.digits_text(crop(img, value_box)))

    def decision(self, img: np.ndarray) -> bool:
        return any("COMMAN" in t.upper() for t in self.ocr_en.readtext(crop(img, COMMAND), detail=0))

    def read(self, frame: np.ndarray) -> Reading:
        img = normalize(frame)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return Reading(message=self.message(img),
                       opp=self.hud(img, hsv, OPP_NAME, OPP_BAR, OPP_PCT),
                       me=self.hud(img, hsv, MY_NAME, MY_BAR, MY_HP, offset=MY_BAR_OFFSET),
                       decision=self.decision(img))
