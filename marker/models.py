from surya.fast_layout import FastLayoutPredictor
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor


def create_model_dict(
    device=None,
    dtype=None,
    attention_implementation: str | None = None,
    inference_manager: SuryaInferenceManager | None = None,
    inference_backend: str | None = None,
) -> dict:
    """Build the predictor set marker uses.

    Two converter modes share this set:
      - balanced (default, GPU): the VLM ``layout_model`` + full-page
        ``recognition_model``.
      - fast (CPU): the lightweight rf-detr/onnx ``fast_layout_model`` + block-mode
        OCR, OCRing only garbled/empty content.

    Table recognition uses the fast rf-detr/onnx ``table_rec_model`` in both
    modes; ``recognition_model`` supplies HTML for scanned tables. The VLM
    ``inference_manager`` is lazy - it only spawns a server when OCR is actually
    needed (so clean digital docs in fast mode never start it). ``device``/
    ``dtype`` apply only to the small torch ocr-error model.
    """
    manager = inference_manager or SuryaInferenceManager(method=inference_backend)
    return {
        "inference_manager": manager,
        "layout_model": LayoutPredictor(manager),
        "fast_layout_model": FastLayoutPredictor(),
        "recognition_model": RecognitionPredictor(manager),
        "table_rec_model": TableRecPredictor(),
        "ocr_error_model": OCRErrorPredictor(
            device=device,
            dtype=dtype,
            attention_implementation=attention_implementation,
        ),
    }


def shutdown_models(model_dict: dict) -> None:
    """Stop the shared inference server if this process spawned it."""
    manager = model_dict.get("inference_manager")
    if manager is not None:
        manager.stop()
