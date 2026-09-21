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
import traceback
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

LOG_FORMAT = "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"


def configure_logging(verbose: bool) -> None:
    """Set up root logging.

    The library calls ``logging.basicConfig`` at import time, which is a no-op once the root
    logger has a handler, so this must run before the library is imported (in the CLI callback
    and in every worker).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format=LOG_FORMAT, force=True
    )


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


def _process_tomogram(
    index: int, tomogram: Path, opts: PredictOptions, predictor: Any, device: str
) -> TomogramResult:
    """Predict -> threshold -> write mask -> (fit planes once) -> write outputs -> thickness.

    Never raises for a per-tomogram problem; the failure is recorded in the result and any
    files already written for this tomogram are removed.
    """
    result = TomogramResult(index=index, path=tomogram, device=str(device))
    written: list[Path] = []
    try:
        import mrcfile
        import numpy as np
        from torch_segment_tomogram_boundaries.measure import measure_thickness
        from torch_segment_tomogram_boundaries.postprocess import (
            fit_slab_planes,
            generate_mask_from_planes,
        )

        paths = output_paths(tomogram, opts)
        probs = predictor.predict_probabilities(
            tomogram,
            slab_size=opts.slab_size,
            batch_size=opts.batch_size,
            smoothing_sigma=opts.smoothing_sigma,
        )
        with mrcfile.open(tomogram, permissive=True, header_only=True) as src:
            voxel_size = src.voxel_size.copy()
        binary = probs > opts.threshold

        def write(key: str, data: Any) -> None:
            mrcfile.write(
                paths[key], data.astype(np.float32), voxel_size=voxel_size, overwrite=True
            )
            written.append(paths[key])

        write("mask", binary)
        result.mask_path = paths["mask"]

        # Fit the top/bottom planes once; reuse them for the fitted mask and thickness.
        planes = None
        if opts.fit_planes or opts.measure_thickness:
            try:
                planes = fit_slab_planes(
                    binary.astype(np.uint8), opts.downsample_grid_size, device=device
                )
            except (ValueError, RuntimeError) as e:
                result.warnings.append(f"plane fitting failed ({e})")
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
        result.thickness_row = None
        for p in written:
            p.unlink(missing_ok=True)
        result.mask_path = None
    finally:
        # Also frees GPU memory held by a failed (e.g. out-of-memory) attempt.
        _release_memory(device)
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
    checkpoint: Path,
    compile_model: bool,
    verbose: bool,
    worker_setup: Optional[Callable[[], None]],
) -> None:
    """Claim one device and build this worker's predictor (once, reused for every tomogram)."""
    configure_logging(verbose)
    device = device_queue.get(timeout=60)
    _WORKER["device"] = device
    try:
        _WORKER["predictor"] = _build_predictor(checkpoint, compile_model, device, worker_setup)
    except Exception as e:  # noqa: BLE001
        # Raising here would break the whole pool with an uninformative error; keep it and
        # report it against every tomogram this worker is handed.
        _WORKER["init_error"] = (f"{type(e).__name__}: {e}", traceback.format_exc())


def _worker_task(index: int, tomogram: Path, opts: PredictOptions) -> TomogramResult:
    device = _WORKER.get("device", "?")
    if "init_error" in _WORKER:
        error, tb = _WORKER["init_error"]
        return TomogramResult(
            index=index, path=tomogram, device=device,
            error=f"worker failed to start on {device}: {error}", traceback=tb,
        )
    return _process_tomogram(index, tomogram, opts, _WORKER["predictor"], device)


# --------------------------------------------------------------------------- public entry point


def run_predictions(
    tasks: Sequence[Path],
    devices: Sequence[str],
    jobs_per_device: int,
    opts: PredictOptions,
    checkpoint: Path,
    compile_model: bool = False,
    verbose: bool = False,
    worker_setup: Optional[Callable[[], None]] = None,
) -> Iterator[TomogramResult]:
    """Predict every tomogram in ``tasks`` and yield one result per tomogram as it completes.

    ``len(devices) * jobs_per_device`` workers are used (never more than there are tomograms),
    each pinned to one device. A single worker runs in-process; otherwise a spawn process pool
    is used, because CUDA cannot be used in forked children. Idle workers pull the next
    tomogram, so uneven tomogram sizes balance automatically. Failures are reported in the
    yielded results rather than raised.

    ``worker_setup`` is an optional top-level (picklable) callable run in each worker before the
    predictor is built; it exists mainly for tests.
    """
    if not tasks:
        return
    slots = assign_devices(devices, jobs_per_device)
    n_workers = min(len(slots), len(tasks))

    if n_workers == 1:
        device = slots[0]
        predictor = _build_predictor(checkpoint, compile_model, device, worker_setup)
        for index, tomogram in enumerate(tasks):
            yield _process_tomogram(index, tomogram, opts, predictor, device)
        return

    ctx = multiprocessing.get_context("spawn")
    device_queue = ctx.Queue()
    for device in slots[:n_workers]:
        device_queue.put(device)

    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(device_queue, checkpoint, compile_model, verbose, worker_setup),
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
        device_queue.close()
