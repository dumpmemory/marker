import pytest
from bs4 import BeautifulSoup

from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.processors.table import TableProcessor


@pytest.mark.config({"page_range": [5]})
def test_table_processor(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    assert len(tables) == 2
    for block in tables:
        # Tables now carry HTML directly (pdftext recon or OCR), no cell blocks.
        assert block.html
        assert "<table" in block.html

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "Schedule" in table_output.markdown


@pytest.mark.filename("table_ex.pdf")
@pytest.mark.config({"page_range": [0], "force_ocr": True})
def test_avoid_double_ocr(pdf_document, recognition_model):
    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    lines = tables[0].contained_blocks(pdf_document, (BlockTypes.Line,))
    assert len(lines) == 0

    processor = TableProcessor(recognition_model, config={"force_ocr": True})
    processor(pdf_document)

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "Participants" in table_output.markdown


@pytest.mark.filename("multicol-blocks.pdf")
@pytest.mark.config({"page_range": [3]})
def test_overlap_blocks(pdf_document, recognition_model):
    page = pdf_document.pages[0]
    assert "Cascading, and the Auxiliary Problem Principle" in page.raw_text(
        pdf_document
    )

    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    assert "Cascading, and the Auxiliary Problem Principle" in page.raw_text(
        pdf_document
    )


@pytest.mark.filename("pres.pdf")
@pytest.mark.config({"page_range": [4]})
def test_ocr_table(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "1.2E-38" in table_output.markdown


@pytest.mark.config({"page_range": [11]})
def test_split_rows(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    table = pdf_document.contained_blocks((BlockTypes.Table,))[-1]
    assert table.html
    rows = BeautifulSoup(table.html, "html.parser").find_all("tr")
    assert len(rows) >= 5
