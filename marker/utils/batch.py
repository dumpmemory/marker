import math

import psutil

from surya.settings import settings as surya_settings


def get_worker_count(oversubscribe: float = 1.5) -> int:
    """Worker processes for batch conversion. The VLM server handles its own
    parallelism, so workers are bounded by CPU work (pdftext, rendering,
    postprocessing) and by how many concurrent requests keep the server
    saturated."""
    physical_cores = psutil.cpu_count(logical=False) or 4
    server_parallel = surya_settings.SURYA_INFERENCE_PARALLEL

    workers = min(
        max(1, physical_cores - 2),
        math.ceil(server_parallel * oversubscribe),
    )
    return workers
