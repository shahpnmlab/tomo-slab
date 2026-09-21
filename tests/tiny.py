"""Helpers that shrink the library's model and add a fake predictor.

They live in an importable module (not conftest) because spawned worker processes have to
run them too: they are passed to ``run_predictions(worker_setup=...)``.
"""
import numpy as np


def tilted_slab(shape=(96, 128, 128), thickness=20.0, slope=0.5):
    """Slab whose surfaces tilt along X; true perpendicular thickness is given."""
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    z0 = 20 + slope * xx
    dz = thickness * np.sqrt(1 + slope**2)  # vertical extent for a perpendicular thickness
    return ((zz >= z0) & (zz <= z0 + dz)).astype(np.float32)


def apply_tiny_settings():
    """Shrink the U-Net and target volume so CPU tests are fast (idempotent)."""
    from torch_segment_tomogram_boundaries import config, constants

    config.MODEL_CONFIG = {
        **config.MODEL_CONFIG,
        "channels": (4, 8, 16, 32),
        "num_res_units": 1,
    }
    constants.TARGET_VOLUME_SHAPE = (16, 64, 64)


def use_fake_predictor():
    """Replace the model by a synthetic slab; tomograms named ``bad*`` raise."""
    import logging
    import warnings

    import mrcfile
    from torch_segment_tomogram_boundaries import predict

    def init(self, checkpoint, compile_model=True, device=None):
        self.device = device

    def predict_probabilities(self, tomo, **kwargs):
        if tomo.name.startswith("bad"):
            raise RuntimeError(f"simulated failure on {tomo.name}")
        logging.info("fake predict %s", tomo.name)
        warnings.warn(f"fake warning for {tomo.name}", stacklevel=2)
        with mrcfile.open(tomo, permissive=True, header_only=True) as mrc:
            shape = (int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx))
        return tilted_slab(shape, thickness=20.0, slope=0.3)

    predict.TomoSlabPredictor.__init__ = init
    predict.TomoSlabPredictor.predict_probabilities = predict_probabilities
