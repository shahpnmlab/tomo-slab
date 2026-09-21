import mrcfile
import numpy as np
import pytest

from tiny import apply_tiny_settings


@pytest.fixture(scope="session", autouse=True)
def tiny_model_settings():
    apply_tiny_settings()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, tiny_model_settings):
    """A real, lightweight checkpoint (untrained tiny U-Net) saved through Lightning.

    Not made by training, so the tests do not depend on the library's data loading.
    """
    import pytorch_lightning as pl
    from torch_segment_tomogram_boundaries import config, constants
    from torch_segment_tomogram_boundaries.losses import get_loss_function
    from torch_segment_tomogram_boundaries.models import create_unet
    from torch_segment_tomogram_boundaries.pl_model import SegmentationModel

    model = SegmentationModel(
        model=create_unet(**config.MODEL_CONFIG),
        loss_function=get_loss_function(config.LOSS_CONFIG),
        learning_rate=1e-4,
        target_shape=constants.TARGET_VOLUME_SHAPE,
    )
    trainer = pl.Trainer(
        accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    trainer.strategy.connect(model)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.ckpt"
    trainer.save_checkpoint(path)
    return path


@pytest.fixture
def make_tomogram(tmp_path):
    def make(name, shape=(24, 48, 48), voxel_size=10.0):
        path = tmp_path / name
        with mrcfile.new(path) as mrc:
            mrc.set_data(np.random.default_rng(0).normal(size=shape).astype(np.float32))
            mrc.voxel_size = voxel_size
        return path

    return make
