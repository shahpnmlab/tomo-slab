import csv

import mrcfile
import numpy as np
import pytest
from typer.testing import CliRunner

from tiny import tilted_slab
from tomo_slab.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "tomo-slab" in result.output


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("prepare", "train", "predict", "fit-planes", "thickness", "fetch"):
        assert cmd in result.output


@pytest.mark.parametrize(
    "cmd", ["prepare", "train", "predict", "fit-planes", "thickness", "fetch"]
)
def test_subcommand_help(cmd):
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0, result.output


def test_positional_metavars_are_explicit():
    assert "INPUT_MASK" in runner.invoke(app, ["fit-planes", "--help"]).output
    assert "OUTPUT_MASK" in runner.invoke(app, ["fit-planes", "--help"]).output
    assert "TOMOGRAMS..." in runner.invoke(app, ["predict", "--help"]).output
    assert "MASKS..." in runner.invoke(app, ["thickness", "--help"]).output


def test_predict_has_no_thickness_flag_but_has_device_options():
    out = runner.invoke(app, ["predict", "--help"]).output
    assert "--thickness-file" in out
    assert "--devices" in out
    assert "--jobs-per-device" in out
    assert "--thickness " not in out


def test_predict_rejects_even_slab_size(make_tomogram):
    tomo = make_tomogram("t.mrc")
    result = runner.invoke(app, ["predict", str(tomo), "--slab-size", "4"])
    assert result.exit_code != 0


def test_predict_rejects_bad_device(make_tomogram):
    tomo = make_tomogram("t.mrc")
    result = runner.invoke(app, ["predict", str(tomo), "--devices", "not-a-device"])
    assert result.exit_code != 0


def test_fit_planes_roundtrip(tmp_path):
    mask = np.zeros((64, 128, 128), np.float32)
    mask[20:44] = 1
    src, dst = tmp_path / "m.mrc", tmp_path / "out" / "fit.mrc"
    mrcfile.write(src, mask)
    result = runner.invoke(app, ["fit-planes", str(src), str(dst)])
    assert result.exit_code == 0, result.output
    with mrcfile.open(dst) as mrc:
        assert mrc.data.shape == mask.shape


def test_thickness_is_perpendicular_not_along_z(tmp_path):
    src, out = tmp_path / "m.mrc", tmp_path / "t.csv"
    with mrcfile.new(src) as mrc:
        mrc.set_data(tilted_slab())
        mrc.voxel_size = 10.0
    result = runner.invoke(app, ["thickness", str(src), "-o", str(out)])
    assert result.exit_code == 0, result.output
    with open(out) as f:
        row = next(csv.DictReader(f))
    # Along-Z would give ~22.4 vox; perpendicular should be ~20 vox = ~20 nm.
    assert abs(float(row["median_vox"]) - 20) < 1.5, row
    assert abs(float(row["median_nm"]) - 20) < 1.5, row
