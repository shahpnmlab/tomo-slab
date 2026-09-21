import csv
import functools

import mrcfile
import numpy as np
import pytest
from typer.testing import CliRunner

from tiny import apply_tiny_settings, use_fake_predictor
from tomo_slab import runner
from tomo_slab.cli import app
from tomo_slab.runner import PredictOptions, assign_devices, parse_devices, run_predictions

cli = CliRunner()


# ------------------------------------------------------------------ device assignment


def test_assign_devices_one_device_per_worker():
    slots = assign_devices(["cuda:0", "cuda:1"], 2)
    assert len(slots) == 4
    assert sorted(slots) == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]


def test_assign_devices_interleaves_so_a_capped_pool_uses_every_device():
    assert assign_devices(["cuda:0", "cuda:1"], 2)[:2] == ["cuda:0", "cuda:1"]


def test_assign_devices_rejects_zero_jobs():
    with pytest.raises(ValueError):
        assign_devices(["cpu"], 0)


def test_parse_devices():
    assert parse_devices("cuda:0, cuda:1") == ["cuda:0", "cuda:1"]
    assert parse_devices("cpu,cpu") == ["cpu", "cpu"]
    with pytest.raises(ValueError):
        parse_devices("cuda:0,,cuda:1")


def test_validate_devices():
    runner.validate_devices(["cpu", "cpu"])
    with pytest.raises(ValueError):
        runner.validate_devices(["nonsense"])
    with pytest.raises(ValueError):
        runner.validate_devices(["cuda:99"])


# ------------------------------------------------------------------ real tiny model, 2 CPU workers


def test_pool_with_real_tiny_checkpoint_and_a_corrupt_tomogram(
    tmp_path, tiny_checkpoint, make_tomogram
):
    tomos = [make_tomogram(f"t{i}.mrc") for i in range(3)]
    corrupt = tmp_path / "corrupt.mrc"
    corrupt.write_bytes(b"this is not an mrc file" * 100)
    tasks = [tomos[0], corrupt, tomos[1], tomos[2]]

    out = tmp_path / "out"
    out.mkdir()
    opts = PredictOptions(output_dir=out, slab_size=3, batch_size=4, save_probabilities=True)
    events = []
    results = list(
        run_predictions(
            tasks, ["cpu", "cpu"], 1, opts, tiny_checkpoint,
            on_progress=events.append, worker_setup=apply_tiny_settings,
        )
    )

    assert len(results) == 4
    by_name = {r.path.name: r for r in results}
    assert not by_name["corrupt.mrc"].ok
    assert by_name["corrupt.mrc"].error
    for t in tomos:
        assert by_name[t.name].ok, by_name[t.name].traceback
        assert (out / f"{t.stem}_mask.mrc").exists()
        assert (out / f"{t.stem}_probabilities.mrc").exists()
        with mrcfile.open(out / f"{t.stem}_mask.mrc") as m:
            assert m.data.shape == (24, 48, 48)
            assert float(m.voxel_size.x) == pytest.approx(10.0)
    # nothing is left behind for the failed tomogram
    assert not list(out.glob("corrupt*"))

    # Progress: every tomogram starts and finishes, and the library's per-slice tqdm loop is
    # reported as measurable progress (this is what the shim is for).
    for t in [*tomos, corrupt]:
        mine = [e for e in events if e.name == t.name]
        assert mine[0].kind == "start" and mine[-1].kind == "finish", mine
    for t in tomos:
        bars = [e for e in events if e.name == t.name and e.total]
        assert {e.phase for e in bars} == {"Slab Blending (XZ axis)", "Slab Blending (YZ axis)"}
        assert max(e.completed for e in bars) == max(e.total for e in bars)


def test_single_worker_runs_in_process(tmp_path, tiny_checkpoint, make_tomogram):
    tomo = make_tomogram("t.mrc")
    opts = PredictOptions(output_dir=tmp_path, slab_size=3, batch_size=4)
    (result,) = run_predictions([tomo], ["cpu"], 1, opts, tiny_checkpoint)
    assert result.ok, result.traceback
    assert (tmp_path / "t_mask.mrc").exists()


def test_worker_startup_failure_is_reported_per_tomogram(tmp_path, make_tomogram):
    tomos = [make_tomogram(f"t{i}.mrc") for i in range(2)]
    opts = PredictOptions(output_dir=tmp_path)
    results = list(
        run_predictions(tomos, ["cpu", "cpu"], 1, opts, tmp_path / "missing.ckpt")
    )
    assert len(results) == 2
    assert all(not r.ok and "failed to start" in r.error for r in results)


# ------------------------------------------------------------------ CLI end to end, fake predictor


@pytest.fixture
def fake_predictor(monkeypatch):
    """Route the CLI through worker processes that use a synthetic tilted slab."""
    from torch_segment_tomogram_boundaries import predict

    # Register the originals so the in-process patch made by `use_fake_predictor` is undone.
    for attr in ("__init__", "predict_probabilities"):
        original = getattr(predict.TomoSlabPredictor, attr)
        monkeypatch.setattr(predict.TomoSlabPredictor, attr, original)
    monkeypatch.setattr(
        runner,
        "run_predictions",
        functools.partial(runner.run_predictions, worker_setup=use_fake_predictor),
    )


def _run_cli(tmp_path, tomos, devices, *extra):
    out = tmp_path / f"out_{devices.replace(',', '_').replace(':', '')}"
    csv_path = out / "thickness.csv"
    ckpt = tmp_path / "unused.ckpt"
    ckpt.write_bytes(b"")
    result = cli.invoke(
        app,
        ["predict", *map(str, tomos), "-c", str(ckpt), "-o", str(out), "--fit-planes",
         "--thickness-file", str(csv_path), "--devices", devices, *extra],
    )
    return result, out, csv_path


def _rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def test_cli_predict_two_cpu_workers(tmp_path, make_tomogram, fake_predictor):
    tomos = [make_tomogram(f"t{i}.mrc", shape=(96, 128, 128)) for i in range(3)]
    result, out, csv_path = _run_cli(tmp_path, tomos, "cpu,cpu")
    assert result.exit_code == 0, result.output
    for t in tomos:
        assert (out / f"{t.stem}_mask.mrc").exists()
        assert (out / f"{t.stem}_fitted_mask.mrc").exists()
    rows = _rows(csv_path)
    assert [r["name"] for r in rows] == [t.name for t in tomos]  # input order
    for r in rows:
        assert abs(float(r["median_nm"]) - 20) < 2, r


def test_cli_multi_worker_matches_single_process(tmp_path, make_tomogram, fake_predictor):
    tomos = [make_tomogram(f"t{i}.mrc", shape=(96, 128, 128)) for i in range(3)]
    r1, out1, csv1 = _run_cli(tmp_path, tomos, "cpu")
    r2, out2, csv2 = _run_cli(tmp_path, tomos, "cpu,cpu", "--jobs-per-device", "2")
    assert r1.exit_code == 0 and r2.exit_code == 0, (r1.output, r2.output)
    for t in tomos:
        for suffix in ("mask", "fitted_mask"):
            with mrcfile.open(out1 / f"{t.stem}_{suffix}.mrc") as a, mrcfile.open(
                out2 / f"{t.stem}_{suffix}.mrc"
            ) as b:
                np.testing.assert_array_equal(a.data, b.data)
    assert _rows(csv1) == _rows(csv2)


def test_cli_failing_tomogram_does_not_abort_the_others(tmp_path, make_tomogram, fake_predictor):
    shape = (96, 128, 128)
    good1, bad, good2 = (make_tomogram(n, shape=shape) for n in ("a.mrc", "bad.mrc", "c.mrc"))
    for devices in ("cpu", "cpu,cpu"):
        result, out, csv_path = _run_cli(tmp_path, [good1, bad, good2], devices)
        assert result.exit_code == 1, result.output
        assert "FAILED bad.mrc" in result.output
        assert "simulated failure" in result.output
        assert (out / "a_mask.mrc").exists() and (out / "c_mask.mrc").exists()
        assert not (out / "bad_mask.mrc").exists()
        assert [r["name"] for r in _rows(csv_path)] == ["a.mrc", "c.mrc"]


def test_cli_overwrite_skips_existing_outputs(tmp_path, make_tomogram, fake_predictor):
    tomo = make_tomogram("t.mrc", shape=(96, 128, 128))
    result, out, _ = _run_cli(tmp_path, [tomo], "cpu")
    assert result.exit_code == 0, result.output
    again = cli.invoke(
        app,
        ["predict", str(tomo), "-c", str(tmp_path / "unused.ckpt"), "-o", str(out),
         "--devices", "cpu"],
    )
    assert again.exit_code == 0
    assert "Skipping t.mrc" in again.output


# ------------------------------------------------------------------ logging and progress display


def test_predict_logs_to_file_and_keeps_the_console_quiet(
    tmp_path, make_tomogram, fake_predictor
):
    shape = (96, 128, 128)
    tomos = [make_tomogram(f"t{i}.mrc", shape=shape) for i in range(2)]
    for devices in ("cpu", "cpu,cpu"):  # in-process, then worker processes
        result, out, _ = _run_cli(tmp_path, tomos, devices)
        assert result.exit_code == 0, result.output
        log = (out / "tomo-slab.log").read_text()
        for t in tomos:
            assert f"fake predict {t.name}" in log
            assert f"fake warning for {t.name}" in log  # Python warnings end up in the log too
        assert "INFO" not in result.output
        assert "fake predict" not in result.output
        assert "fake warning" not in result.output
        assert "logging to" in result.output


def test_predict_log_file_option(tmp_path, make_tomogram, fake_predictor):
    tomo = make_tomogram("t.mrc", shape=(96, 128, 128))
    log = tmp_path / "custom" / "my.log"
    result, out, _ = _run_cli(tmp_path, [tomo], "cpu", "--log-file", str(log))
    assert result.exit_code == 0, result.output
    assert "fake predict t.mrc" in log.read_text()
    assert not (out / "tomo-slab.log").exists()


def test_run_predictions_reports_progress_from_workers(tmp_path, make_tomogram, fake_predictor):
    from tomo_slab.runner import ProgressEvent

    tomos = [make_tomogram(f"t{i}.mrc", shape=(96, 128, 128)) for i in range(3)]
    opts = PredictOptions(output_dir=tmp_path, fit_planes=True)
    events = []
    results = list(
        run_predictions(
            tomos, ["cpu", "cpu"], 1, opts, tmp_path / "unused.ckpt",
            on_progress=events.append, worker_setup=use_fake_predictor,
        )
    )
    assert all(r.ok for r in results)
    assert all(isinstance(e, ProgressEvent) for e in events)
    for t in tomos:
        phases = [e.phase for e in events if e.name == t.name]
        assert phases[0] == "loading"
        assert "fitting planes" in phases


def test_progress_view_tracks_rows_and_ignores_late_events():
    from tomo_slab.progress import ProgressView
    from tomo_slab.runner import ProgressEvent

    with ProgressView(2) as view:
        view.handle(ProgressEvent("start", 0, "a.mrc", "cuda:0", "loading"))
        view.handle(ProgressEvent("update", 0, "a.mrc", "cuda:0", "Slab Blending (XZ axis)", 5, 10))
        view.handle(ProgressEvent("start", 1, "b [x].mrc", "cuda:1", "loading"))  # markup-ish name
        assert set(view._rows) == {0, 1}
        view.handle(ProgressEvent("finish", 0, "a.mrc", "cuda:0"))
        view.complete(0)
        view.handle(ProgressEvent("update", 0, "a.mrc", "cuda:0", "late", 9, 10))
        assert set(view._rows) == {1}
        view.complete(1)
        assert view._done == 2
