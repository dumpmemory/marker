import pytest

from marker.processors.document_toc import DocumentTOCProcessor


@pytest.mark.config({"page_range": [0]})
def test_document_toc_processor(pdf_document, recognition_model):
    processor = DocumentTOCProcessor()
    processor(pdf_document)

    # Page 0 has exactly: title, Abstract, 1. Introduction
    assert len(pdf_document.table_of_contents) == 3
    assert pdf_document.table_of_contents[0]["title"] == "Subspace Adversarial Training"
