import pytest

from marker.schema import BlockTypes


@pytest.mark.config({"page_range": [0]})
@pytest.mark.filename("adversarial_rot.pdf")
def test_rotated_bboxes(pdf_document):
    first_page = pdf_document.pages[0]

    text_blocks = first_page.contained_blocks(
        pdf_document, (BlockTypes.Text, BlockTypes.TextInlineMath)
    )
    assert len(text_blocks) > 0

    # Rotated pages fail the embedded-text checks and get block-level OCR
    text_lines = first_page.contained_blocks(pdf_document, (BlockTypes.Line,))
    if text_lines:
        # Embedded text was used - line bboxes must stay within layout blocks
        max_line_position = max([line.polygon.x_end for line in text_lines])
        max_block_position = max(
            [block.polygon.x_end for block in text_blocks if block.source == "layout"]
        )
        assert max_line_position <= max_block_position
    else:
        # OCR'd - blocks should carry html and stay within the page bounds
        ocr_blocks = [b for b in text_blocks if b.html]
        assert len(ocr_blocks) > 0
        assert all(
            b.polygon.x_end <= first_page.polygon.x_end * 1.02 for b in text_blocks
        )
