#!/usr/bin/env python
"""docling throughput + scoreable output (run in docling's own venv).

docling parallelizes inside a single process (models loaded once):
``convert_all()`` streams many documents and
``settings.perf.{doc_batch_concurrency,page_batch_concurrency}`` control how
many run at once (both default to 1 -> single stream). We raise those and pin
the accelerator, then stream the whole olmOCR-bench set.

Device note: a default ``pip install docling`` pulls CPU-only PyTorch, so
``--device auto`` runs on CPU. Install a CUDA build of torch in the docling
venv to use ``--device cuda``.

Writes ``<out>/<category>/<stem>_pg1_repeat1.md`` for scoring and prints
sustained pages/sec.
"""

import argparse
import glob
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--out", default="", help="write markdown here for scoring")
    ap.add_argument("--source-subdir", default="pdfs")
    ap.add_argument("--doc-concurrency", type=int, default=os.cpu_count())
    ap.add_argument("--page-concurrency", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--limit", type=int, default=0, help="cap number of PDFs (debug/quick rate)"
    )
    args = ap.parse_args()

    from docling.datamodel.settings import settings

    settings.perf.doc_batch_concurrency = args.doc_concurrency
    settings.perf.doc_batch_size = args.doc_concurrency
    settings.perf.page_batch_concurrency = args.page_concurrency
    settings.inference.compile_torch_models = False  # keep warmup out of timing

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    device = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
    }[args.device]
    popts = PdfPipelineOptions()
    popts.accelerator_options = AcceleratorOptions(
        num_threads=args.threads, device=device
    )
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=popts)}
    )

    pdf_dir = os.path.join(args.bench_dir, args.source_subdir)
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    if args.limit:
        pdfs = pdfs[: args.limit + 5]  # +5 for the warmup slice
    print(
        f"docling: {len(pdfs)} pdfs, doc_conc={args.doc_concurrency} "
        f"page_conc={args.page_concurrency} device={args.device}",
        flush=True,
    )

    # Warm up on a few docs so lazy model load is excluded from the timing.
    for _ in conv.convert_all(pdfs[:5], raises_on_error=False):
        pass

    # Map PDF stem -> bench-relative path, so results land in the right
    # category subdir regardless of how docling names its internal document.
    stem_to_rel = {
        os.path.splitext(os.path.basename(p))[0]: os.path.relpath(p, pdf_dir)
        for p in pdfs
    }

    body = pdfs[5:]
    t0 = time.monotonic()
    ok = 0
    for res in conv.convert_all(body, raises_on_error=False):
        try:
            md = res.document.export_to_markdown()
        except Exception:
            continue
        ok += 1
        if not args.out:
            continue
        try:
            stem = os.path.splitext(os.path.basename(str(res.input.file)))[0]
            rel = stem_to_rel.get(stem, stem)  # category/stem.pdf
            base = os.path.splitext(os.path.basename(rel))[0]
            d = os.path.join(args.out, os.path.dirname(rel))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{base}_pg1_repeat1.md"), "w") as f:
                f.write(md)
        except Exception:
            pass  # output is best-effort; throughput is the primary metric
    dt = time.monotonic() - t0
    print(
        f"DONE docling: {ok}/{len(body)} in {dt:.1f}s | "
        f"throughput {ok / dt:.2f} pages/s "
        f"({dt / len(body) * 1000:.1f} ms/page effective)",
        flush=True,
    )


if __name__ == "__main__":
    main()
