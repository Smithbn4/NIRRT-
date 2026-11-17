"""Lazy 2D map/A* loader; same semantic channels and labels as the original."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.grid_map import load_binary_map, network_input, normalize_input, pad_to_32


class ImageDataset(Dataset):
    def __init__(self, data_root, mode='train', path_thickness=3,
                 augment=False, limit=None):
        if path_thickness < 1 or path_thickness % 2 != 1:
            raise ValueError('path thickness must be a positive odd number')
        self.root = Path(data_root).expanduser().resolve()/mode
        self.path_thickness, self.augment = path_thickness, augment
        with (self.root/'envs.json').open() as stream:
            environments = json.load(stream)
        self.samples = []
        for env_index, env in enumerate(environments):
            for query_index, (start, goal) in enumerate(zip(env['start'], env['goal'])):
                self.samples.append((env_index, query_index, start, goal, tuple(env['env_dims'])))
        if limit is not None:
            if limit < 1:
                raise ValueError('sample limit must be positive')
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError('No samples in {}'.format(self.root))
        shapes = {sample[-1] for sample in self.samples}
        if len(shapes) != 1:
            raise ValueError('Use maps of one resolution per dataset for batched training')

    def __len__(self):
        return len(self.samples)

    def label_for(self, sample):
        env_index, query_index, _, _, shape = sample
        filename = self.root/'astar_paths'/'{}_{}.txt'.format(env_index, query_index)
        path = np.atleast_2d(np.loadtxt(str(filename), delimiter=',')).astype(int)
        height, width = shape
        if path.shape[1] != 2 or np.any(path < 0) or np.any(path[:, 0] >= width) or np.any(path[:, 1] >= height):
            raise ValueError('Invalid path coordinates in {}'.format(filename))
        label = np.zeros(shape, dtype=np.uint8)
        label[path[:, 1], path[:, 0]] = 1
        return cv2.dilate(label, np.ones((self.path_thickness, self.path_thickness), np.uint8))

    def class_weights(self):
        counts = np.zeros(2, dtype=np.float64)
        for sample in self.samples:
            label = self.label_for(sample)
            positives = label.sum()
            counts += [label.size-positives, positives]
        return np.power(counts.max()/np.maximum(counts, 1), 1/3).astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        env_index, query_index, start, goal, shape = sample
        mask = load_binary_map(self.root/'env_imgs'/'{}.png'.format(env_index))
        if mask.shape != shape:
            raise ValueError('Image dimensions disagree with envs.json')
        raw = network_input(mask, start, goal)
        label = self.label_for(sample).astype(np.int64)
        if self.augment:
            if np.random.random() < 0.5:
                raw, label = np.flip(raw, -1), np.flip(label, -1)
            if np.random.random() < 0.5:
                raw, label = np.flip(raw, -2), np.flip(label, -2)
            turns = np.random.randint(4) if shape[0] == shape[1] else 2*np.random.randint(2)
            raw, label = np.rot90(raw, turns, axes=(-2, -1)), np.rot90(label, turns)
        raw = normalize_input(pad_to_32(raw))
        label = pad_to_32(label, fill=-100)  # exclude padded pixels from loss/metrics
        return torch.from_numpy(raw.copy()), torch.from_numpy(label.copy()), '{}_{}'.format(env_index, query_index)

