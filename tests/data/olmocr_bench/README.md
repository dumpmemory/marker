# Vendored olmOCR-bench pages

Three single-page PDFs from [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
(allenai, Apache-2.0), with the subset of that benchmark's own tests that apply
to them. Used by `tests/converters/test_olmocr_bench.py` as an end-to-end quality
integration test: marker converts each page and the vendored rules must pass.

| local file | source (olmOCR-bench `pdfs/`) | tests |
|---|---|---|
| `pdfs/multi_column_page1.pdf` | `multi_column/0005784d0d255f6652180433936fa2998188_page_1_pg1.pdf` | 5 × order (reading order) |
| `pdfs/long_tiny_text_pg36.pdf` | `long_tiny_text/11_pg36_pg1.pdf` | 4 × present (small-text OCR) |
| `pdfs/headers_footers_page1.pdf` | `headers_footers/4b91e05fb0fe865391f0f25c41a83c9c5fd37c08_page_1.pdf` | 4 × absent (header/footer stripping) + 1 × baseline |

`tests.jsonl` holds one record per test; each record's `source` field records the
original benchmark PDF path. Only the `present`/`absent`/`order`/`baseline` rule
types are vendored — the `table`/`math` types need olmOCR-bench's own KaTeX +
table-parsing checker and aren't reimplemented here.

The pages were chosen because current marker passes every one of their tests in
balanced mode; this is a regression guard, not a score reproduction. To refresh
or expand the set, run the real harness in `benchmarks/` against full olmOCR-bench.
