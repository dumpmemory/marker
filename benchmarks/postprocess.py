"""Normalize marker markdown for olmOCR-bench scoring (output-only).

olmOCR-bench matches on literal text, so output that is *equivalent* but encoded
differently silently fails tests: HTML `<sub>`/`<sup>`, backslash-escaped
markdown (`\\_`, `\\*`), HTML-escaped characters inside math spans, LaTeX left in
table cells, and image alt-text. These transforms map that output to what the
checker expects — the same normalization Chandra's official olmOCR-bench run
applies to its own markdown. Applied by default in ``inference.py``; disable
with ``--raw``.

Net effect is small (marker's markdown is already fairly clean): on olmOCR-bench
it lifts balanced ~+0.3 overall (mostly arxiv_math + tables), fast ~+0.1, and is
within noise for fast-no-OCR — but it is the correct, apples-to-apples way to
score against a checker other VLM-OCR systems are also normalized for.
"""

import html as _html
import re

try:
    import unicodeit
except Exception:  # optional; only used for LaTeX-in-table-cell conversion
    unicodeit = None

_SUP = {c: u for c, u in zip("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")}
_SUB = {c: u for c, u in zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")}


def _latex_to_unicode(s: str) -> str:
    """LaTeX -> Unicode for table-cell math (Greek, symbols, ^/_)."""
    s = _html.unescape(s)
    s = re.sub(r"\\(?:text|mathrm)\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    if unicodeit is not None:
        try:
            s = unicodeit.replace(s)
        except Exception:
            pass
    s = (
        re.sub(r"\\[a-zA-Z]+", "", s)
        .replace("{", "")
        .replace("}", "")
        .replace("\\", "")
    )
    return re.sub(r"\s+", " ", s).strip()


def _subsup_to_unicode(s: str) -> str:
    s = re.sub(
        r"<sub>(.*?)</sub>",
        lambda m: "".join(_SUB.get(c, c) for c in m.group(1)),
        s,
        flags=re.S | re.I,
    )
    s = re.sub(
        r"<sup>(.*?)</sup>",
        lambda m: "".join(_SUP.get(c, c) for c in m.group(1)),
        s,
        flags=re.S | re.I,
    )
    return re.sub(r"</?su[bp]>", "", s)


def _strip_escapes_outside_math(s: str) -> str:
    parts = re.split(r"(\$\$.*?\$\$|\$[^$\n]+\$)", s, flags=re.S)
    for i in range(0, len(parts), 2):  # even indices are non-math
        parts[i] = re.sub(r"\\([_*$%&#.+()\[\]!>~^{}])", r"\1", parts[i])
    return "".join(parts)


def postprocess(md: str) -> str:
    """Map marker markdown to olmOCR-bench-friendly text (output-only)."""
    s = re.sub(
        r"<math\b[^>]*>(.*?)</math>",
        lambda m: _latex_to_unicode(m.group(1)),
        md,
        flags=re.S | re.I,
    )
    s = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: "$$" + _html.unescape(m.group(1)) + "$$",
        s,
        flags=re.S,
    )
    s = re.sub(r"\$([^$\n]+)\$", lambda m: "$" + _html.unescape(m.group(1)) + "$", s)
    s = _subsup_to_unicode(s)
    s = _strip_escapes_outside_math(s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)  # drop image markdown / alt-text
    return s
