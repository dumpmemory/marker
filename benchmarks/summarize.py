#!/usr/bin/env python
"""Turn olmOCR-bench per-category pass rates into the numbers we report.

olmOCR-bench's checker prints a pass rate per test category. Our tables report
two derived numbers:

  * **Overall**      = macro-average over all 8 categories (how olmOCR-bench and
    Chandra report the headline score).
  * **Digital-only** = macro-average over the 6 non-scanned categories, i.e.
    excluding ``old_scans`` and ``old_scans_math``.

Feed it the per-category scores (as printed by the checker), either as a JSON
file mapping category -> pass rate, or as ``name=value`` args:

    python benchmarks/summarize.py \
        arxiv_math=82.0 old_scans_math=63.8 tables=72.8 old_scans=43.3 \
        long_tiny_text=71.3 multi_column=77.0 headers_footers=95.8 baseline=99.7

    python benchmarks/summarize.py --json scores.json

Values may be 0-1 or 0-100; the scale is detected and results print as 0-100.
"""

import argparse
import json
import sys

SCANNED = {"old_scans", "old_scans_math"}


def _norm(name):
    return name.replace(".jsonl", "").strip()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pairs", nargs="*", help="category=score pairs")
    ap.add_argument("--json", help="JSON file mapping category -> score")
    args = ap.parse_args()

    scores = {}
    if args.json:
        scores.update(
            {_norm(k): float(v) for k, v in json.load(open(args.json)).items()}
        )
    for p in args.pairs:
        if "=" not in p:
            sys.exit(f"bad pair (expected name=value): {p}")
        k, v = p.split("=", 1)
        scores[_norm(k)] = float(v)
    if not scores:
        sys.exit("no scores given (use name=value args or --json)")

    # Detect 0-1 vs 0-100 and normalize to 0-100.
    if max(scores.values()) <= 1.0:
        scores = {k: v * 100 for k, v in scores.items()}

    digital = {k: v for k, v in scores.items() if k not in SCANNED}
    overall = sum(scores.values()) / len(scores)
    dig = sum(digital.values()) / len(digital)

    width = max(len(k) for k in scores)
    for k in sorted(scores):
        tag = "" if k not in SCANNED else "  (scanned)"
        print(f"  {k:<{width}}  {scores[k]:6.1f}{tag}")
    print("-" * (width + 12))
    print(
        f"  {'Overall (macro, ' + str(len(scores)) + ' cats)':<{width}}  {overall:6.1f}"
    )
    print(f"  {'Digital-only (' + str(len(digital)) + ' cats)':<{width}}  {dig:6.1f}")


if __name__ == "__main__":
    main()
