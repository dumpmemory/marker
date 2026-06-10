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
    # Layout, recognition, and table rec share a single VLM server managed by
    # the inference manager (spawned lazily on first use, or attached via
    # SURYA_INFERENCE_URL / the on-disk sentinel). device/dtype only apply to
    # the small torch-based ocr error model.
    manager = inference_manager or SuryaInferenceManager(method=inference_backend)
    return {
        "inference_manager": manager,
        "layout_model": LayoutPredictor(manager),
        "recognition_model": RecognitionPredictor(manager),
        "table_rec_model": TableRecPredictor(manager),
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
