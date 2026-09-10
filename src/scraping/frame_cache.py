"""Build frames on file-arrival events, independently of network polling."""
from datetime import datetime, timedelta


def run_frame_cache_worker(file_ready_queue, model_ready_queue=None, build_frame=None):
    if build_frame is None:
        from data_processing.data_loading import build_and_cache_frame
        build_frame = build_and_cache_frame

    pending = set()
    completed = set()
    newest = None
    while True:
        tick = file_ready_queue.get()
        if tick is None:  # Allows orderly shutdown and deterministic tests.
            return
        tick = int(tick)
        newest = max(newest or tick, tick)
        cutoff = int((datetime.strptime(str(newest), "%Y%m%d%H%M") - timedelta(minutes=20)).strftime("%Y%m%d%H%M"))
        completed = {item for item in completed if item >= cutoff}
        pending = {item for item in pending if item >= cutoff}
        if tick >= cutoff and tick not in completed:
            pending.add(tick)
        for item in sorted(pending.copy()):
            try:
                result = build_frame(item, img_name="70km", verbose=False)
            except Exception as exc:
                print(f"Cache attempt failed for {item}: {exc}")
                continue
            if result is not None:
                pending.remove(item)
                completed.add(item)
                print(f"Multimodal frame ready for {item}")
                if model_ready_queue is not None:
                    model_ready_queue.put(item)
