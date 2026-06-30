import pytest

from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.processors.table import TableProcessor
from marker.schema import BlockTypes


@pytest.mark.filename("water_damage.pdf")
def test_garbled_pdf(pdf_document, recognition_model, table_rec_model):
    tables = pdf_document.pages[0].contained_blocks(pdf_document, (BlockTypes.Table,))
    assert len(tables) > 0

    # Garbled page - the table goes through full-mode OCR table rec
    processor = TableProcessor(recognition_model, table_rec_model)
    processor(pdf_document)

    table = pdf_document.pages[0].contained_blocks(pdf_document, (BlockTypes.Table,))[0]
    assert table.html and "<table" in table.html
    assert "варіант" in table.raw_text(pdf_document)


@pytest.mark.filename("hindi_judgement.pdf")
@pytest.mark.config({"page_range": [2, 3], "disable_ocr": True})
def test_garbled_builder(config, doc_provider, ocr_error_model):
    line_builder = LineBuilder(ocr_error_model, config)
    builder = DocumentBuilder(config)
    document = builder.build_document(doc_provider)

    bad_ocr_results = line_builder.ocr_error_detection(
        document.pages, doc_provider.page_lines
    )
    assert len(bad_ocr_results.labels) == 2
    assert any([label == "bad" for label in bad_ocr_results.labels])


@pytest.mark.filename("adversarial.pdf")
@pytest.mark.config({"page_range": [2, 3], "disable_ocr": True})
def test_nongarbled_builder(config, doc_provider, ocr_error_model):
    line_builder = LineBuilder(ocr_error_model, config)
    builder = DocumentBuilder(config)
    document = builder.build_document(doc_provider)

    bad_ocr_results = line_builder.ocr_error_detection(
        document.pages, doc_provider.page_lines
    )
    assert len(bad_ocr_results.labels) == 2
    assert all([label == "good" for label in bad_ocr_results.labels])
