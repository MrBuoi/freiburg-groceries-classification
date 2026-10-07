"""Project configuration shared by training, evaluation, and notebooks."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
SPLIT_PATH = ARTIFACT_DIR / "split_manifest.json"

SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 25
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RESNET_LEARNING_RATE = 1e-4
RESNET_WARMUP_EPOCHS = 2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CNN_MEAN = (0.5, 0.5, 0.5)
CNN_STD = (0.5, 0.5, 0.5)


def get_device():
    """Select the fastest available local PyTorch device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
