"""
A 5 x 7 pixel font (printable ASCII), drawn for this tool, to write ROI names into the QC
pictures without an extra package. Umlauts and ß are written out (ä -> ae); other
characters outside ASCII become '?'.
"""
from __future__ import annotations

import numpy as np

WIDTH, HEIGHT = 5, 7
# 7 rows per character (ASCII 32-126 in order), each row a base-32 digit of 5 pixel bits
_ROWS = (
    "00000004444404aaa0000aavavaa4fke5u4op248j3cik8lid44800002488842842224804lel40044v4400000c48000v000"
    "00000cc01248g0ehjlphe4c4444eeh1248vv2421he26aiv22vgu11he68guhhev124888ehhehheehhf12c0cc0cc00cc0c48"
    "248g84200v0v008421248eh12404eh1dlleehhhvhhuhhuhhuehggghesihhhisvgguggvvggugggehgnhhfhhhvhhhe44444e"
    "72222ichikokihggggggvhrllhhhhhpljhhehhhhheuhhugggehhhliduhhukihfgge11uv444444hhhhhhehhhhha4hhhllla"
    "hha4ahhhhha444v1248gve88888e0g84210e22222e4ah0000000000v842000000e1fhfggmphhu00egghe11djhhf00ehvge"
    "698s8880fhhf1eggmphhh40c444e20622icggikokic44444e00qllhh00mphhh00ehhhe00uhugg00djf1100mpggg00ege1u"
    "88s889600hhhjd00hhha400hhlla00ha4ah00hhf1e00v248v244844244444448442448008l200"
)
_DIGITS = "0123456789abcdefghijklmnopqrstuv"
_TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}


def _glyph(ch: str) -> np.ndarray:
    code = ord(ch) if 32 <= ord(ch) <= 126 else ord("?")
    rows = _ROWS[(code - 32) * HEIGHT:(code - 31) * HEIGHT]
    return np.array([[(_DIGITS.index(r) >> (WIDTH - 1 - b)) & 1 for b in range(WIDTH)] for r in rows], bool)


def ascii_text(text: str) -> str:
    return "".join(_TRANSLIT.get(c, c) for c in text)


def render(text: str, scale: int = 2) -> np.ndarray:
    """The text as a boolean image (one pixel of space between characters), scaled up."""
    text = ascii_text(text)
    out = np.zeros((HEIGHT, max(1, len(text) * (WIDTH + 1) - 1)), bool)
    for i, ch in enumerate(text):
        out[:, i * (WIDTH + 1):i * (WIDTH + 1) + WIDTH] = _glyph(ch)
    return np.kron(out, np.ones((scale, scale), bool))
