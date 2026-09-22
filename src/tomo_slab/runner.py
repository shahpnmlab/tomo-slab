"""Run tomogram predictions on one or more devices.

The unit of work is one tomogram. With more than one worker, a spawn-based process pool is
used: every worker is pinned to one device, loads the model once, and then pulls tomograms
until none are left. With a single worker everything runs in-process.

Heavy dependencies (torch, the library) are imported lazily so importing this module is cheap.
"""
from __future__ import annotations

import gc
import logging
import multiprocessing
import queue
import threading
import traceback
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

LOG_FORMAT = "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"


def configure_logging(verbose: bool, log_file: Optional[Path] = None) -> None:
    """Set up root logging: to the console, or to ``log_file`` only if one is given.

    With a log file the console stays quiet (the CLI draws progress bars there) and Python
    warnings are routed into the log as well. The library calls ``logging.basicConfig`` at
    import time, which is a no-op once the root logger has a handler, so this must run before
    the library is imported (in the CLI and in every worker).
    """
    level = logging.DEBUG if verbose else logging.INFO
    if log_file is None:
        logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
        return
    # Every process appends to the same file; one short line per write keeps them intact.
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.captureWarnings(False)  # captureWarnings(True) is a no-op if already on; reset first
    logging.captureWarnings(True)


@dataclass(frozen=True)
class ProgressEvent:
    """Progress of one tomogram, sent from a worker to whoever draws the progress bars.

    ``kind`` is ``"start"``, ``"update"`` or ``"finish"``. ``slot`` numbers the worker, which
    is what owns a progress bar. ``total`` is None while the current ``phase`` has no
    measurable progress.
    """

    kind: str
    index: int
    name: str
    device: str
    phase: str = ""
    completed: int = 0
    total: Optional[int] = None
    slot: int = 0


# --------------------------------------------------------------------------- devices


def parse_devices(spec: str) -> list[str]:
    """Split a comma-separated device list such as ``"cuda:0, cuda:1"``."""
    devices = [d.strip() for d in spec.split(",")]
    if not all(devices):
        raise ValueError(f"empty device name in {spec!r}")
    return devices


def default_devices() -> list[str]:
    """All visible CUDA devices, else ``["cpu"]``."""
    import torch

    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]


def validate_devices(devices: Sequence[str]) -> None:
    """Raise ``ValueError`` for malformed or unavailable devices."""
    import torch

    for name in devices:
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"invalid device {name!r}: {e}") from e
        if device.type == "cuda":
            n = torch.cuda.device_count()
            if device.index is None:
                raise ValueError(f"{name!r}: give an explicit index, e.g. cuda:0")
            if device.index >= n:
                raise ValueError(f"{name!r} is not available ({n} CUDA device(s) visible)")


def assign_devices(devices: Sequence[str], jobs_per_device: int) -> list[str]:
    """One device string per worker: ``devices * jobs_per_device``, interleaved.

    Interleaving (``[cuda:0, cuda:1, cuda:0, cuda:1]``) means that a pool capped to fewer
    workers than slots still spreads over all devices.
    """
    if jobs_per_device < 1:
        raise ValueError("jobs_per_device must be >= 1")
    return [d for _ in range(jobs_per_device) for d in devices]


def worker_devices(devices: Sequence[str], jobs_per_device: int, n_tasks: int) -> list[str]:
    """The device of each worker that will run: one per slot, but no more than tasks."""
    return assign_devices(devices, jobs_per_device)[:n_tasks]


# --------------------------------------------------------------------------- data classes


@dataclass(frozen=True)
class PredictOptions:
    """Everything a worker needs besides the tomogram itself (paths and plain values only)."""

    output_dir: Path
    threshold: float = 0.5
    slab_size: int = 15
    batch_size: int = 16
    smoothing_sigma: Optional[float] = None
    save_probabilities: bool = False
    fit_planes: bool = False
    downsample_grid_size: int = 8
    measure_thickness: bool = False
    parallel_axes: bool = False


@dataclass
class TomogramResult:
    """Outcome for one tomogram."""

    index: int
    path: Path
    device: str = ""
    mask_path: Optional[Path] = None
    thickness_row: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None  # one-line summary of the failure
    traceback: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def output_paths(tomogram: Path, opts: PredictOptions) -> dict[str, Path]:
    """Output files that ``opts`` asks for, for one tomogram."""
    stem = tomogram.stem
    paths = {"mask": opts.output_dir / f"{stem}_mask.mrc"}
    if opts.save_probabilities:
        paths["probabilities"] = opts.output_dir / f"{stem}_probabilities.mrc"
    if opts.fit_planes:
        paths["fitted_mask"] = opts.output_dir / f"{stem}_fitted_mask.mrc"
    return paths


# --------------------------------------------------------------------------- the actual work


def _release_memory(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda"):
        import torch

        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _reset_gpu_peaks(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(device)


def _log_gpu_peaks(name: str, device: str) -> None:
    """Log this process's peak GPU memory for the tomogram just processed.

    ``allocated`` is memory held by live tensors; ``reserved`` is what the caching allocator
    took from the driver (what nvidia-smi shows), which can be higher, notably with several
    CUDA streams because each keeps its own cache of free blocks.
    """
    if str(device).startswith("cuda"):
        import torch

        gib = 1024**3
        logging.info(
            "%s: peak GPU memory on %s: %.2f GiB allocated, %.2f GiB reserved",
            name, device, torch.cuda.max_memory_allocated(device) / gib,
            torch.cuda.max_memory_reserved(device) / gib,
        )


# Which tomogram this process is working on, and where to report progress (or None).
_CURRENT: dict[str, Any] = {}


def _emit(kind: str, phase: str = "", completed: int = 0, total: Optional[int] = None) -> None:
    report = _CURRENT.get("report")
    if report is not None:
        report(
            ProgressEvent(
                kind, _CURRENT["index"], _CURRENT["name"], _CURRENT["device"],
                phase, completed, total, _CURRENT["slot"],
            )
        )


# The library's tqdm loops running at the moment: id -> [completed, total, finished]. With
# concurrent axes (see `_predict_axes_in_parallel`) there are two, reported as one bar.
_LOOPS: dict[int, list] = {}
_LOOPS_LOCK = threading.Lock()


class _ProgressTqdm:
    """Stand-in for ``tqdm`` inside the library's predict module.

    The library draws a tqdm bar over the slices of each axis. Several workers doing that at
    once garble the terminal, so the loop reports to ``_emit`` instead of drawing anything.
    Loops that run at the same time are combined into a single bar.
    """

    def __init__(self, iterable, desc="", total=None, **_ignored):
        self.iterable, self.desc = iterable, desc
        self.total = total if total is not None else len(iterable)

    def _report(self, completed: int, finished: bool = False) -> None:
        with _LOOPS_LOCK:
            _LOOPS[id(self)] = [completed, self.total, finished]
            if len(_LOOPS) == 1:
                desc = self.desc
            else:
                desc = "Slab Blending (XZ+YZ)"
            _emit(
                "update", desc,
                sum(loop[0] for loop in _LOOPS.values()),
                sum(loop[1] for loop in _LOOPS.values()),
            )
            if all(loop[2] for loop in _LOOPS.values()):
                _LOOPS.clear()  # start afresh with the next axis (or tomogram)

    def __iter__(self):
        step = max(1, self.total // 200)  # ~200 events per loop is plenty
        self._report(0, finished=self.total <= 0)
        n = 0
        for n, item in enumerate(self.iterable, 1):
            yield item
            if n % step == 0 or n >= self.total:
                self._report(n, finished=n >= self.total)
        if n < self.total:  # the iterable ended early
            self._report(n, finished=True)


def _install_progress_shim() -> Optional[Any]:
    """Swap the library's tqdm for ``_ProgressTqdm``; returns the original to restore."""
    from torch_segment_tomogram_boundaries import predict

    original = getattr(predict, "tqdm", None)
    if original is not None:
        predict.tqdm = _ProgressTqdm
    return original


def _restore_progress(original: Optional[Any]) -> None:
    if original is not None:
        from torch_segment_tomogram_boundaries import predict

        predict.tqdm = original


def _predict_probabilities(
    predictor: Any, tomogram: Path, slab_size: int, batch_size: int,
    smoothing_sigma: Optional[float],
) -> Any:
    return predictor.predict_probabilities(
        tomogram, slab_size=slab_size, batch_size=batch_size, smoothing_sigma=smoothing_sigma
    )


def _predict_axes_in_parallel(
    predictor: Any, tomogram: Path, slab_size: int, batch_size: int,
    smoothing_sigma: Optional[float],
) -> Any:
    """`TomoSlabPredictor.predict_probabilities`, but with the XZ and YZ passes run together.

    The tomogram is loaded and uploaded once and both passes read it. Each pass runs in its own
    thread and, on CUDA, on its own stream, so the GPU can overlap them; that pays off when one
    pass alone leaves the GPU partly idle (small batches, launch-bound slices), at the price of
    a second set of activations in VRAM. The model is shared, which is fine for eval-mode
    inference, but not for ``torch.compile(mode="reduce-overhead")`` (CUDA graphs).

    This mirrors the body of the library's method (it has no way to run the axes concurrently),
    so keep the two in step.
    """
    import mrcfile
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_segment_tomogram_boundaries.utils import threeD

    device = predictor.device
    on_cuda = device.type == "cuda"

    with mrcfile.open(tomogram, permissive=True) as mrc:
        data = mrc.data.astype(np.float32)
    original_shape = data.shape
    logging.info("Input tomogram shape: %s", original_shape)

    with torch.no_grad():
        volume = threeD.resize_and_pad_3d(
            torch.from_numpy(data), target_shape=predictor.target_shape_3d, mode="image"
        ).to(device)
        del data
        if on_cuda:
            torch.cuda.synchronize(device)  # the side streams must see the uploaded volume

        def run_axis(axis: str, view: Any) -> Any:
            if not on_cuda:
                return predictor._predict_single_axis_with_slab_blending(
                    view, axis, slab_size, batch_size
                )
            stream = torch.cuda.Stream(device)
            with torch.cuda.stream(stream):
                out = predictor._predict_single_axis_with_slab_blending(
                    view, axis, slab_size, batch_size
                )
            stream.synchronize()
            return out

        logging.info("Predicting along the XZ and YZ axes at the same time.")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="axis") as pool:
            xz = pool.submit(run_axis, "XZ", volume)
            yz = pool.submit(run_axis, "YZ", volume.permute(0, 2, 1))
            pred_xz, pred_yz = xz.result(), yz.result().permute(0, 2, 1)
        if on_cuda:  # made on side streams, used from now on on the current one
            for pred in (pred_xz, pred_yz):
                pred.record_stream(torch.cuda.current_stream(device))

        logging.info("Averaging predictions from both axes.")
        prob_map = (pred_xz + pred_yz) / 2.0
        del pred_xz, pred_yz, volume

        if smoothing_sigma and smoothing_sigma > 0:
            logging.info("Applying 3D Gaussian smoothing with sigma=%s...", smoothing_sigma)
            prob_map = threeD.gpu_gaussian_blur_3d(prob_map, smoothing_sigma, device)

        logging.info("Resizing prediction back to original shape %s...", original_shape)
        return (
            F.interpolate(prob_map[None, None], size=original_shape, mode="area")
            .squeeze()
            .cpu()
            .numpy()
        )


def _process_tomogram(
    index: int,
    tomogram: Path,
    opts: PredictOptions,
    predictor: Any,
    device: str,
    report: Optional[Callable[[ProgressEvent], None]] = None,
    slot: int = 0,
) -> TomogramResult:
    """Predict -> threshold -> write mask -> (fit planes once) -> write outputs -> thickness.

    Never raises for a per-tomogram problem; the failure is recorded in the result and any
    files already written for this tomogram are removed.
    """
    result = TomogramResult(index=index, path=tomogram, device=str(device))
    written: list[Path] = []
    _CURRENT.update(
        report=report, index=index, name=tomogram.name, device=str(device), slot=slot
    )
    _LOOPS.clear()
    _reset_gpu_peaks(device)
    _emit("start", "loading")
    try:
        import mrcfile
        import numpy as np
        from torch_segment_tomogram_boundaries.measure import measure_thickness
        from torch_segment_tomogram_boundaries.postprocess import (
            fit_slab_planes,
            generate_mask_from_planes,
        )

        paths = output_paths(tomogram, opts)
        logging.info("%s: predicting on %s", tomogram.name, device)
        predict = _predict_axes_in_parallel if opts.parallel_axes else _predict_probabilities
        probs = predict(
            predictor, tomogram, opts.slab_size, opts.batch_size, opts.smoothing_sigma
        )
        with mrcfile.open(tomogram, permissive=True, header_only=True) as src:
            voxel_size = src.voxel_size.copy()
        binary = probs > opts.threshold

        def write(key: str, data: Any) -> None:
            mrcfile.write(
                paths[key], data.astype(np.float32), voxel_size=voxel_size, overwrite=True
            )
            written.append(paths[key])

        _emit("update", "writing mask")
        write("mask", binary)
        result.mask_path = paths["mask"]

        # Fit the top/bottom planes once; reuse them for the fitted mask and thickness.
        planes = None
        if opts.fit_planes or opts.measure_thickness:
            _emit("update", "fitting planes")
            try:
                planes = fit_slab_planes(
                    binary.astype(np.uint8), opts.downsample_grid_size, device=device
                )
            except (ValueError, RuntimeError) as e:
                result.warnings.append(f"plane fitting failed ({e})")
                logging.warning("%s: plane fitting failed (%s)", tomogram.name, e)
        if opts.measure_thickness:
            result.thickness_row = {
                "name": tomogram.name,
                **measure_thickness(binary, float(voxel_size.x), planes=planes),
            }
        if opts.fit_planes and planes is not None:
            write("fitted_mask", generate_mask_from_planes(planes, binary.shape))
        if opts.save_probabilities:
            write("probabilities", probs)
    except Exception as e:  # noqa: BLE001 - one bad tomogram must not stop the others
        result.error = f"{type(e).__name__}: {e}"
        result.traceback = traceback.format_exc()
        logging.error("%s failed:\n%s", tomogram.name, result.traceback)
        result.thickness_row = None
        for p in written:
            p.unlink(missing_ok=True)
        result.mask_path = None
    finally:
        _log_gpu_peaks(tomogram.name, device)
        # Also frees GPU memory held by a failed (e.g. out-of-memory) attempt.
        _release_memory(device)
        _emit("finish")
        _CURRENT.clear()
    return result


# --------------------------------------------------------------------------- worker process

# Per-worker state, set once by `_init_worker`.
_WORKER: dict[str, Any] = {}


def _build_predictor(
    checkpoint: Path, compile_model: bool, device: str, worker_setup: Optional[Callable[[], None]]
) -> Any:
    if worker_setup is not None:
        worker_setup()
    from torch_segment_tomogram_boundaries.predict import TomoSlabPredictor

    return TomoSlabPredictor(checkpoint, compile_model=compile_model, device=device)


def _init_worker(
    device_queue: Any,
    progress_queue: Any,
    checkpoint: Path,
    compile_model: bool,
    verbose: bool,
    log_file: Optional[Path],
    worker_setup: Optional[Callable[[], None]],
) -> None:
    """Claim a (slot, device) pair and build this worker's predictor (reused for each tomogram)."""
    configure_logging(verbose, log_file)
    slot, device = device_queue.get(timeout=60)
    _WORKER["slot"], _WORKER["device"] = slot, device
    try:
        _WORKER["predictor"] = _build_predictor(checkpoint, compile_model, device, worker_setup)
        if progress_queue is not None:
            _WORKER["report"] = progress_queue.put
            _install_progress_shim()
    except Exception as e:  # noqa: BLE001
        # Raising here would break the whole pool with an uninformative error; keep it and
        # report it against every tomogram this worker is handed.
        logging.error("worker on %s failed to start:\n%s", device, traceback.format_exc())
        _WORKER["init_error"] = (f"{type(e).__name__}: {e}", traceback.format_exc())


def _worker_task(index: int, tomogram: Path, opts: PredictOptions) -> TomogramResult:
    device = _WORKER.get("device", "?")
    if "init_error" in _WORKER:
        error, tb = _WORKER["init_error"]
        return TomogramResult(
            index=index, path=tomogram, device=device,
            error=f"worker failed to start on {device}: {error}", traceback=tb,
        )
    return _process_tomogram(
        index, tomogram, opts, _WORKER["predictor"], device, _WORKER.get("report"),
        _WORKER["slot"],
    )


def _drain(
    events: Any, on_progress: Callable[[ProgressEvent], None], stop: threading.Event
) -> None:
    """Forward events from the workers' queue to ``on_progress`` until told to stop."""
    while True:
        try:
            event = events.get(timeout=0.1)
        except queue.Empty:
            if stop.is_set():
                return
            continue
        try:
            on_progress(event)
        except Exception:  # noqa: BLE001 - a broken progress display must not stop the run
            logging.exception("progress callback failed")


# --------------------------------------------------------------------------- public entry point


def run_predictions(
    tasks: Sequence[Path],
    devices: Sequence[str],
    jobs_per_device: int,
    opts: PredictOptions,
    checkpoint: Path,
    compile_model: bool = False,
    verbose: bool = False,
    log_file: Optional[Path] = None,
    on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    worker_setup: Optional[Callable[[], None]] = None,
) -> Iterator[TomogramResult]:
    """Predict every tomogram in ``tasks`` and yield one result per tomogram as it completes.

    ``len(devices) * jobs_per_device`` workers are used (never more than there are tomograms;
    see `worker_devices`), each pinned to one device and numbered by `ProgressEvent.slot`. A
    single worker runs in-process; otherwise a spawn process pool is used, because CUDA cannot
    be used in forked children. Idle workers pull the next tomogram, so uneven tomogram sizes
    balance automatically. Failures are reported in the yielded results rather than raised.

    ``log_file`` is the file the workers log to (see `configure_logging`; the caller sets up
    logging in this process). If ``on_progress`` is given it is called with `ProgressEvent`s
    as work advances, and the library's own tqdm bars are silenced. It is called from a
    background thread when there are several workers.

    ``worker_setup`` is an optional top-level (picklable) callable run in each worker before the
    predictor is built; it exists mainly for tests.
    """
    if not tasks:
        return
    slots = worker_devices(devices, jobs_per_device, len(tasks))
    n_workers = len(slots)

    if n_workers == 1:
        device = slots[0]
        predictor = _build_predictor(checkpoint, compile_model, device, worker_setup)
        original_tqdm = _install_progress_shim() if on_progress is not None else None
        try:
            for index, tomogram in enumerate(tasks):
                yield _process_tomogram(
                    index, tomogram, opts, predictor, device, on_progress, slot=0
                )
        finally:
            _restore_progress(original_tqdm)
        return

    ctx = multiprocessing.get_context("spawn")
    device_queue = ctx.Queue()
    for slot, device in enumerate(slots):
        device_queue.put((slot, device))
    progress_queue = ctx.Queue() if on_progress is not None else None
    stop = threading.Event()
    drain_thread = None
    if progress_queue is not None:
        drain_thread = threading.Thread(
            target=_drain, args=(progress_queue, on_progress, stop), daemon=True
        )
        drain_thread.start()

    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            device_queue, progress_queue, checkpoint, compile_model, verbose, log_file,
            worker_setup,
        ),
    )
    futures: dict[Future, tuple[int, Path]] = {}
    try:
        for index, tomogram in enumerate(tasks):
            futures[executor.submit(_worker_task, index, tomogram, opts)] = (index, tomogram)
        for future in as_completed(futures):
            index, tomogram = futures[future]
            try:
                yield future.result()
            except BrokenProcessPool:
                # A worker died hard (e.g. killed by the OS); each unfinished future fails.
                yield TomogramResult(
                    index=index, path=tomogram, error="worker process died unexpectedly"
                )
            except Exception as e:  # noqa: BLE001 - e.g. an unpicklable result
                yield TomogramResult(
                    index=index, path=tomogram, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(),
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        # Workers have exited, so every event is already in the pipe; let the thread drain it.
        stop.set()
        if drain_thread is not None:
            drain_thread.join(timeout=5)
        device_queue.close()
