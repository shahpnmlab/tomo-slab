# tomo-slab

Command-line tool for segmenting the top/bottom boundaries of a slab (e.g. a lamella) in cryo-ET
tomograms. It is a thin wrapper over the [`torch-tomo-slab`](https://github.com/teamtomo/torch-segment-tomogram-boundaries)
library, plus multi-GPU job distribution for `predict`.

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

## Development

```bash
uv venv && uv sync
uv run pytest
```
