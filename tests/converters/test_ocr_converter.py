import pytest

from marker.converters.ocr import OCRConverter
from marker.renderers.ocr_json import OCRJSONOutput, OCRJSONPageOutput


def _ocr_converter(config, model_dict, temp_pdf, eq_count: int):
    converter = OCRConverter(artifact_dict=model_dict, config=config)

    ocr_json: OCRJSONOutput = converter(temp_pdf.name)
    pages = ocr_json.children

    assert len(pages) == 1
    eqs = [block for block in pages[0].children if block.block_type == "Equation"]
    assert len(eqs) == eq_count
    return pages


def check_bboxes(page: OCRJSONPageOutput, blocks):
    page_size = page.bbox
    for block in blocks:
        assert block.html
        bbox = block.bbox
        assert all(
            [
                bbox[0] >= page_size[0],
                bbox[1] >= page_size[1],
                bbox[2] <= page_size[2],
                bbox[3] <= page_size[3],
            ]
        ), "Block bbox is outside page bbox"


@pytest.mark.config({"page_range": [0]})
def test_ocr_converter(config, model_dict, temp_doc):
    _ocr_converter(config, model_dict, temp_doc, 2)


@pytest.mark.filename("pres.pdf")
@pytest.mark.config({"page_range": [1], "force_ocr": True})
def test_ocr_converter_force(config, model_dict, temp_doc):
    pages = _ocr_converter(config, model_dict, temp_doc, 0)
    blocks = [block for block in pages[0].children if block.block_type != "Equation"]
    assert len(blocks) > 0
    check_bboxes(pages[0], blocks)
