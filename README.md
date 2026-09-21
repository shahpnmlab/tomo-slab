# tomo-slab

Command-line tool for segmenting the top/bottom boundaries of a slab (e.g. a lamella) in cryo-ET
tomograms. It is a thin wrapper over the [`torch-segment-tomogram-boundaries`](https://github.com/teamtomo/torch-segment-tomogram-boundaries)
library, plus multi-GPU job distribution for `predict`.

## Installation

`tomo-slab` and the `torch-tomo-slab` library it wraps (repository `torch-segment-tomogram-boundaries`) are not on PyPI yet, so install from GitHub.

**With [uv](https://docs.astral.sh/uv/) (recommended)** — installs `tomo-slab` as an isolated tool:

```bash
uv tool install git+https://github.com/shahpnmlab/tomo-slab
tomo-slab --version
```

To update to the latest version, run `uv tool upgrade tomo-slab` (re-running `uv tool install` does
nothing once the tool is installed; add `--reinstall` if an upgrade does not take effect).

**With pip** — pip does not know where to find the library, so give both as git URLs:

```bash
python -m venv .venv && source .venv/bin/activate
pip install \
  "torch-tomo-slab @ git+https://github.com/teamtomo/torch-segment-tomogram-boundaries@v0.1.0" \
  "tomo-slab @ git+https://github.com/shahpnmlab/tomo-slab"
```

Multi-GPU prediction needs a CUDA build of PyTorch that matches your driver; see
[pytorch.org](https://pytorch.org/get-started/locally/) if `torch.cuda.is_available()` is false.

## Usage

```bash
tomo-slab fetch                                   # download the pretrained checkpoint
tomo-slab predict *.mrc -o out --fit-planes --thickness-file thickness.csv
tomo-slab predict *.mrc --devices cuda:0,cuda:1 --jobs-per-device 2
tomo-slab prepare VOLUME_DIR MASK_DIR -o prepared_data
tomo-slab train prepared_data --ckpt-dir ckpts
tomo-slab fit-planes INPUT_MASK OUTPUT_MASK
tomo-slab thickness MASKS... -o thickness.csv
```

## Multi-GPU `predict`

The unit of work is one tomogram. `--devices` takes comma-separated torch device strings
(default: every visible CUDA device, else `cpu`), and `--jobs-per-device N` starts N worker
processes per device, so there are `len(devices) * N` workers in total. Each worker loads the model
once and pulls tomograms as it becomes free, so tomograms of different sizes balance themselves.

A tomogram that fails (including a CUDA out-of-memory error) is reported and skipped; the remaining
tomograms are still processed, and the exit code is non-zero if any failed.

Every worker holds its own copy of the model and the full-volume tensors, so GPU memory use grows
with `--jobs-per-device`. Tune it to your VRAM and tomogram size.

With one device and `--jobs-per-device 1` everything runs in-process (no worker pool).

While running, `predict` shows one progress row per tomogram being processed. Log messages
(model loading, warnings, failures with tracebacks) are kept off the console and written to
`<output-dir>/tomo-slab.log`, or to the file given with `--log-file`; `-v` adds debug output.

## Development

```bash
uv venv && uv sync
uv run pytest
```
