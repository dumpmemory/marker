import pytest

from marker.schema import BlockTypes


def _ocr_pipeline_test(pdf_document):
    first_page = pdf_document.pages[0]
    assert first_page.text_extraction_method == "surya"
    assert len(first_page.structure) > 0

    # Full-page OCR rebuilds the page structure from the model output
    first_block = first_page.get_block(first_page.structure[0])
    assert first_block.text_extraction_method == "surya"
    assert first_block.block_type == BlockTypes.SectionHeader

    # OCR'd blocks carry the model html directly, with no line children
    assert first_block.structure == []
    assert "Subspace Adversarial Training" in first_block.html

    # No Line blocks are created for OCR'd pages
    text_lines = first_page.contained_blocks(pdf_document, (BlockTypes.Line,))
    assert len(text_lines) == 0

    # Every text-bearing block should have html from the OCR pass
    ocr_blocks = [
        block
        for block in first_page.structure_blocks(pdf_document)
        if block.block_type
        in (BlockTypes.Text, BlockTypes.SectionHeader, BlockTypes.TextInlineMath)
    ]
    assert len(ocr_blocks) > 0
    assert all(block.html for block in ocr_blocks)


@pytest.mark.config({"force_ocr": True, "page_range": [0]})
def test_ocr_pipeline(pdf_document):
    _ocr_pipeline_test(pdf_document)


@pytest.mark.config({"force_ocr": True, "page_range": [0], "use_llm": True})
def test_ocr_with_inline_pipeline(pdf_document):
    _ocr_pipeline_test(pdf_document)
