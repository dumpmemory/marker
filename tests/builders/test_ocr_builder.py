from marker.builders.ocr import OcrBuilder


def test_clean_html(recognition_model):
    builder = OcrBuilder(recognition_model)

    # Debug attributes are stripped, truncated tags are balanced
    html = '<p data-bbox="1 2 3 4" data-label="Text">Hello <b>world'
    cleaned = builder.clean_html(html)
    assert "data-bbox" not in cleaned
    assert "data-label" not in cleaned
    assert "</b>" in cleaned
    assert "Hello" in cleaned

    # Repetition loops are dropped
    looping = "<p>" + "same phrase " * 400
    assert builder.clean_html(looping) == ""

    assert builder.clean_html("") == ""
