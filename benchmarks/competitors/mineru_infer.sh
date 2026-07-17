#!/bin/bash
# MinerU throughput + scoreable output (run with MinerU's own venv on PATH).
#
# MinerU's native concurrency is its pipeline backend: given a directory it
# batches pages across files through each model stage (layout / OCR / formula /
# table). We give it a large virtual-VRAM budget so it forms big GPU batches,
# run it per olmOCR-bench category, time the whole thing, and collect the
# markdown into the scoring layout (<out>/<category>/<stem>_pg1_repeat1.md).
#
# Usage:
#   BENCH_DIR=/path/to/olmOCR-bench/bench_data \
#   OUT=/tmp/mineru MINERU_BIN=/path/to/mineru \
#     bash benchmarks/competitors/mineru_infer.sh
set -eu

BENCH_DIR=${BENCH_DIR:?set BENCH_DIR to olmOCR-bench bench_data}
OUT=${OUT:-/tmp/mineru}
PDF_DIR="$BENCH_DIR/pdfs"
RAW="$OUT/raw"
MINERU_BIN=${MINERU_BIN:-mineru}

export MINERU_DEVICE_MODE=${MINERU_DEVICE_MODE:-cuda}
export MINERU_VIRTUAL_VRAM_SIZE=${MINERU_VIRTUAL_VRAM_SIZE:-70}

rm -rf "$RAW"; mkdir -p "$RAW" "$OUT"

total=0
t0=$(date +%s.%N)
for cat in "$PDF_DIR"/*/; do
  cat=$(basename "$cat")
  n=$(find "$PDF_DIR/$cat" -name '*.pdf' 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && continue
  "$MINERU_BIN" -p "$PDF_DIR/$cat" -o "$RAW/$cat" -b pipeline -m auto \
    > "$OUT/mineru_$cat.log" 2>&1
  total=$((total + n))
  # Collect markdown into the olmOCR-bench scoring layout.
  mkdir -p "$OUT/$cat"
  for md in "$RAW/$cat"/*/auto/*.md; do
    [ -f "$md" ] || continue
    name=$(basename "$md" .md)
    cp "$md" "$OUT/$cat/${name}_pg1_repeat1.md"
  done
  echo "  $cat: n=$n done ($(date +%H:%M:%S))"
done
t1=$(date +%s.%N)

python3 - "$total" "$t0" "$t1" <<'PY'
import sys
total, t0, t1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
dt = t1 - t0
print(f"DONE mineru: {total} pages in {dt:.1f}s | throughput {total/dt:.2f} "
      f"pages/s ({dt/total*1000:.0f} ms/page effective)")
PY
