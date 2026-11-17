"""Paths and runtime settings shared by the standalone 2D project."""
from pathlib import Path
import random
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'data/random_2d'
DEFAULT_CHECKPOINT = ROOT / 'checkpoints/icra2024_resnet50_unet.pt'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name='auto', threads=4):
    import torch
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu or auto')
    return torch.device(name)


def load_model_weights(model, checkpoint, device):
    import torch
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint))
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)

