"""Convert occupancy grids and endpoints into ResNet50/U-Net guidance."""
import numpy as np
import torch
from datasets.grid_map import network_input, normalize_input, pad_to_32
from project_config import DEFAULT_CHECKPOINT, load_model_weights
from unet.models.unet_model import UNet


class GNGWrapper:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device='cpu'):
        self.device = torch.device(device)
        # Full trained weights are supplied, so no ImageNet download is needed.
        self.model = UNet(pretrained=False).to(self.device)
        load_model_weights(self.model, checkpoint, self.device)
        self.model.eval()

    @torch.inference_mode()
    def classify_path_points(self, binary_mask, x_start, x_goal):
        height, width = binary_mask.shape
        raw = pad_to_32(network_input(binary_mask, x_start, x_goal))
        inputs = torch.from_numpy(normalize_input(raw)).unsqueeze(0).to(self.device)
        logits = self.model(inputs)[0, :, :height, :width]
        prediction = logits.argmax(dim=0).cpu().numpy()
        scores = logits.softmax(dim=0)[1].cpu().numpy()
        coordinates = np.flip(np.argwhere(prediction == 1), axis=1).copy()
        return coordinates, prediction, scores

