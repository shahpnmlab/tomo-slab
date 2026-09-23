"""Command-line interface for tomo-slab.

Heavy dependencies (torch, lightning, ...) are imported lazily inside each command
so that ``--help`` and ``--version`` stay fast.
"""
from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from tomo_slab import __version__
from tomo_slab.runner import configure_logging

app = typer.Typer(
    name="tomo-slab",
    help="Segment slab boundaries in tomographic volumes with a 2D U-Net.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)


class Accelerator(str, Enum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tomo-slab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Segment slab boundaries in tomographic volumes."""
    # Must happen before the library is imported: it calls logging.basicConfig() at import.
    configure_logging(verbose)
    ctx.meta["tomo_slab.verbose"] = verbose


def _ascii_histogram(values: list[float], unit: str, bins: int = 10, width: int = 40) -> str:
    """Render ``values`` as a horizontal ASCII histogram binned over their range."""
    import numpy as np

    lo, hi = min(values), max(values)
    edges = np.array([lo, lo + 1.0]) if lo == hi else np.linspace(lo, hi, bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    max_count = counts.max() or 1
    lines = [f"Mean thickness across {len(values)} tomogram(s) ({unit}):"]
    for i, count in enumerate(counts):
        bar = "#" * int(round(width * count / max_count))
        lines.append(f"  [{edges[i]:7.2f}, {edges[i + 1]:7.2f}) | {bar} {count}")
    return "\n".join(lines)


def _report_thickness(rows: list[dict], output: Optional[Path]) -> None:
    """Print a thickness histogram to screen and optionally write the raw rows to a CSV file."""
    from torch_segment_tomogram_boundaries.measure import write_thickness_csv

    failed = [r for r in rows if r["mean_vox"] != r["mean_vox"]]  # NaN -> empty mask
    ok = [r for r in rows if r["mean_vox"] == r["mean_vox"]]
    for r in failed:
        typer.secho(
            f"{r['name']}: could not measure thickness (mask empty or too small to fit planes)",
            fg=typer.colors.YELLOW,
        )
    if ok:
        has_nm = all(r["voxel_size_A"] == r["voxel_size_A"] for r in ok)
        unit = "nm" if has_nm else "vox"
        values = [r["mean_nm"] if has_nm else r["mean_vox"] for r in ok]
        typer.echo(_ascii_histogram(values, unit))
    if output is not None and rows:
        write_thickness_csv(rows, output)
        typer.echo(f"Thickness table written to {output}")


@app.command()
def prepare(
    volume_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, metavar="VOLUME_DIR",
        help="Directory containing the input tomograms (*.mrc).",
    ),
    mask_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, metavar="MASK_DIR",
        help="Directory containing the ground-truth boundary masks (*.mrc). "
        "File names must match those in VOLUME_DIR.",
    ),
    output_dir: Path = typer.Option(
        Path("prepared_data"), "--output-dir", "-o",
        help="Output root; slices go to <output-dir>/train and <output-dir>/val.",
    ),
    validation_fraction: float = typer.Option(
        0.2, "--val-fraction", min=0.0, max=1.0,
        help="Fraction of volumes reserved for validation.",
    ),
) -> None:
    """Convert 3D tomograms + masks into 2D training slices."""
    from torch_segment_tomogram_boundaries.processing import TrainingDataGenerator

    generator = TrainingDataGenerator(
        volume_dir=volume_dir,
        mask_dir=mask_dir,
        output_train_dir=output_dir / "train",
        output_val_dir=output_dir / "val",
        validation_fraction=validation_fraction,
    )
    generator.run()
    typer.secho(f"Prepared data written to {output_dir}", fg=typer.colors.GREEN)


@app.command()
def train(
    data_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, metavar="DATA_DIR",
        help="Prepared data root containing train/ and val/ subdirectories "
        "(the --output-dir of `tomo-slab prepare`).",
    ),
    ckpt_dir: Optional[Path] = typer.Option(
        None, "--ckpt-dir", "-o", help="Where to save checkpoints and logs."
    ),
    learning_rate: Optional[float] = typer.Option(None, "--lr", help="Learning rate."),
    max_epochs: Optional[int] = typer.Option(None, "--epochs", "-e", min=1),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", min=1),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", min=0),
    accelerator: Accelerator = typer.Option(Accelerator.auto, "--accelerator"),
    devices: Optional[int] = typer.Option(None, "--devices", min=1, help="Number of devices."),
) -> None:
    """Train a segmentation model on prepared 2D slices."""
    from torch_segment_tomogram_boundaries import config
    from torch_segment_tomogram_boundaries.trainer import train as run_train

    train_dir, val_dir = data_dir / "train", data_dir / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise typer.BadParameter(f"Expected directory {d} (run `tomo-slab prepare` first).")

    # Settings not exposed by `train()` are read from `config` at setup time.
    if batch_size is not None:
        config.BATCH_SIZE = batch_size
    if num_workers is not None:
        config.NUM_WORKERS = num_workers

    kwargs: dict = {"accelerator": accelerator.value}
    if devices is not None:
        kwargs["devices"] = devices
    run_train(
        train_data_dir=train_dir,
        val_data_dir=val_dir,
        ckpt_save_dir=ckpt_dir or config.CKPT_SAVE_PATH,
        learning_rate=learning_rate if learning_rate is not None else config.LEARNING_RATE,
        max_epochs=max_epochs if max_epochs is not None else config.MAX_EPOCHS,
        **kwargs,
    )


@app.command()
def predict(
    ctx: typer.Context,
    tomograms: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, metavar="TOMOGRAMS...",
        help="One or more tomograms (.mrc) to segment.",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", "-c", exists=True, dir_okay=False,
        help="Model checkpoint. Defaults to the pretrained model (downloaded if needed).",
    ),
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o", file_okay=False, help="Directory for output masks."
    ),
    threshold: float = typer.Option(0.5, "--threshold", "-t", min=0.0, max=1.0),
    slab_size: int = typer.Option(
        15, "--slab-size", min=1, help="Odd slab size for blending; 1 disables."
    ),
    batch_size: int = typer.Option(16, "--batch-size", "-b", min=1),
    smoothing_sigma: Optional[float] = typer.Option(
        None, "--smoothing-sigma", min=0.0, help="3D Gaussian smoothing of probabilities."
    ),
    save_probabilities: bool = typer.Option(
        False, "--save-probabilities", help="Also write the probability map."
    ),
    compile_model: bool = typer.Option(
        False, "--compile/--no-compile", help="Use torch.compile (slower start, faster inference)."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs."),
    downsample_grid_size: int = typer.Option(
        8, "--downsample-grid-size", min=1,
        help="Surface-point downsampling grid used when fitting the mask's top/bottom planes.",
    ),
    thickness_file: Optional[Path] = typer.Option(
        None, "--thickness-file", dir_okay=False,
        help="Measure slab thickness and write it for all tomograms to this CSV file.",
    ),
    devices: Optional[str] = typer.Option(
        None, "--devices", metavar="DEVICES",
        help="Comma-separated torch devices to run on, e.g. cuda:0,cuda:1. "
        "Default: all visible CUDA devices, else cpu.",
    ),
    jobs_per_device: int = typer.Option(
        1, "--jobs-per-device", min=1,
        help="Worker processes per device. Each worker holds its own copy of the model and "
        "full-volume tensors, so tune this to the available VRAM and tomogram size.",
    ),
    parallel_axes: bool = typer.Option(
        False, "--parallel-axes",
        help="Run the XZ and YZ passes of each tomogram at the same time (on separate CUDA "
        "streams), loading the tomogram once. Faster when one pass leaves the GPU partly idle, "
        "but needs more VRAM for the second pass's activations, so it is off by default. "
        "CUDA only; not compatible with --compile.",
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", dir_okay=False,
        help="Where to write log messages (from all workers), which are kept off the console. "
        "Default: <output-dir>/tomo-slab.log.",
    ),
) -> None:
    """Predict slab masks for one or more tomograms, optionally measuring thickness.

    The mask is produced by thresholding the prediction and then fitting planes to its top
    and bottom surfaces, so void regions are cleanly separated from signal; if plane fitting
    fails (e.g. too few boundary points) the raw thresholded mask is saved instead, with a
    warning. One tomogram is one unit of work, and there are one worker process per device per
    --jobs-per-device (a single worker runs in-process). A tomogram that fails is reported and
    skipped; the exit code is non-zero if any failed.
    """  # noqa: E501
    from tomo_slab.progress import ProgressView
    from tomo_slab.runner import (
        PredictOptions,
        configure_logging,
        default_devices,
        output_paths,
        parse_devices,
        run_predictions,
        validate_devices,
        worker_devices,
    )

    if slab_size % 2 == 0:
        raise typer.BadParameter("must be odd", param_hint="--slab-size")

    try:
        device_list = parse_devices(devices) if devices is not None else default_devices()
        validate_devices(device_list)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--devices") from e

    if parallel_axes:
        if compile_model:
            raise typer.BadParameter(
                "the model is shared by both passes, which CUDA graphs cannot do",
                param_hint="--parallel-axes/--compile",
            )
        if not all(d.startswith("cuda") for d in device_list):
            raise typer.BadParameter("needs CUDA devices only", param_hint="--parallel-axes")

    stems: dict[str, Path] = {}
    for tomo in tomograms:
        if tomo.stem in stems and stems[tomo.stem] != tomo:
            raise typer.BadParameter(
                f"{stems[tomo.stem]} and {tomo} would write the same output files",
                param_hint="TOMOGRAMS...",
            )
        stems[tomo.stem] = tomo
    tomograms = list(stems.values())

    opts = PredictOptions(
        output_dir=output_dir,
        threshold=threshold,
        slab_size=slab_size,
        batch_size=batch_size,
        smoothing_sigma=smoothing_sigma,
        save_probabilities=save_probabilities,
        downsample_grid_size=downsample_grid_size,
        measure_thickness=thickness_file is not None,
        parallel_axes=parallel_axes,
    )

    # Honour --overwrite in the parent, before anything is submitted.
    todo: list[Path] = []
    for tomo in tomograms:
        existing = [p for p in output_paths(tomo, opts).values() if p.exists()]
        if existing and not overwrite:
            typer.secho(
                f"Skipping {tomo.name}: {existing[0]} exists (use --overwrite).",
                fg=typer.colors.YELLOW,
            )
        else:
            todo.append(tomo)
    if not todo:
        return

    # Log messages go to a file so that they do not garble the progress bars.
    verbose = bool(ctx.meta.get("tomo_slab.verbose", False))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file or output_dir / "tomo-slab.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(verbose, log_file)
    logging.info("tomo-slab %s: %s", __version__, " ".join(sys.argv))

    if checkpoint is None:  # download once here, not in every worker
        from torch_segment_tomogram_boundaries.fetch import get_latest_checkpoint

        checkpoint = get_latest_checkpoint()

    slots = worker_devices(device_list, jobs_per_device, len(todo))
    typer.echo(
        f"Predicting {len(todo)} tomogram(s) with {len(slots)} worker(s) on "
        f"{','.join(dict.fromkeys(device_list))}; logging to {log_file}"
    )

    thickness_rows: list[tuple[int, dict]] = []
    failed: list[Path] = []
    with ProgressView(slots, [t.name for t in todo]) as view:
        for n_done, result in enumerate(
            run_predictions(
                todo, device_list, jobs_per_device, opts, checkpoint,
                compile_model=compile_model, verbose=verbose, log_file=log_file,
                on_progress=view.handle,
            ),
            1,
        ):
            view.complete(result.index)
            for w in result.warnings:
                view.print(f"{result.path.name}: {w}", style="yellow", err=True)
            if result.ok:
                view.print(
                    f"[{n_done}/{len(todo)}] {result.path.name} -> {result.mask_path} "
                    f"[{result.device}]",
                    style="green",
                )
                if result.thickness_row is not None:
                    thickness_rows.append((result.index, result.thickness_row))
            else:
                failed.append(result.path)
                view.print(
                    f"[{n_done}/{len(todo)}] FAILED {result.path.name}: {result.error}",
                    style="red", err=True,
                )
                if verbose and result.traceback:
                    view.print(result.traceback, err=True)

    if thickness_file is not None:
        # Completion order is arbitrary; report in input order.
        _report_thickness([row for _, row in sorted(thickness_rows, key=lambda x: x[0])],
                          thickness_file)

    if failed:
        typer.secho(
            f"{len(failed)} of {len(todo)} tomogram(s) failed.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)


@app.command("fit-planes")
def fit_planes(
    input_mask: Path = typer.Argument(
        ..., exists=True, dir_okay=False, metavar="INPUT_MASK",
        help="Existing binary mask (.mrc) to refine.",
    ),
    output_mask: Path = typer.Argument(
        ..., dir_okay=False, metavar="OUTPUT_MASK",
        help="Path of the fitted mask (.mrc) to write.",
    ),
    downsample_grid_size: int = typer.Option(
        8, "--downsample-grid-size", "-g", min=1,
        help="Voxel grid size for downsampling surface points before fitting.",
    ),
) -> None:
    """Refine a binary mask by fitting planes to its top and bottom surfaces."""
    import mrcfile
    import numpy as np
    from torch_segment_tomogram_boundaries.postprocess import fit_and_generate_mask

    with mrcfile.open(input_mask, permissive=True) as mrc:
        mask = mrc.data.astype(np.uint8)
        voxel_size = mrc.voxel_size.copy()

    try:
        fitted = fit_and_generate_mask(mask, downsample_grid_size)
    except (ValueError, RuntimeError) as e:
        typer.secho(f"Plane fitting failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    mrcfile.write(output_mask, fitted, voxel_size=voxel_size, overwrite=True)
    typer.secho(f"Fitted mask saved to {output_mask}", fg=typer.colors.GREEN)


@app.command()
def thickness(
    masks: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, metavar="MASKS...",
        help="One or more existing binary masks (.mrc) to measure.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", dir_okay=False, help="Also write results to this CSV file."
    ),
) -> None:
    """Measure slab thickness (perpendicular top-bottom distance) of binary masks."""
    import mrcfile
    from torch_segment_tomogram_boundaries.measure import measure_thickness

    rows = []
    for path in masks:
        with mrcfile.open(path, permissive=True) as mrc:
            rows.append({"name": path.name, **measure_thickness(mrc.data, float(mrc.voxel_size.x))})
    _report_thickness(rows, output)


@app.command()
def fetch(
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", file_okay=False, help="Directory to store the checkpoint."
    ),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Local filename for the checkpoint."
    ),
) -> None:
    """Download the pretrained checkpoint and print its path."""
    from torch_segment_tomogram_boundaries.fetch import get_latest_checkpoint

    typer.echo(get_latest_checkpoint(cache_dir=cache_dir, filename=filename))


if __name__ == "__main__":
    app()
