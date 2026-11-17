"""Evaluate 2D U-Net path segmentation on labeled image maps."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from project_config import DEFAULT_DATA, DEFAULT_CHECKPOINT, select_device, load_model_weights
from unet.image_dataset import ImageDataset
from unet.models.unet_model import UNet
from unet.training import run_epoch
from unet.reporting import plot_loss_curve, save_accuracy_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--output', type=Path, help='Save final accuracy reports here; also plot metrics.csv if present')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    device = select_device(args.device, args.threads)
    train = ImageDataset(args.data_root, 'train')
    test = ImageDataset(args.data_root, args.split, limit=args.max_samples)
    model = UNet(pretrained=False).to(device)
    load_model_weights(model, args.checkpoint, device)
    weights = torch.tensor(train.class_weights(), device=device)
    result = run_epoch(model, DataLoader(test, batch_size=args.batch_size), device, weights)
    print(json.dumps(result, indent=2))
    if args.output is not None:
        output = args.output.expanduser().resolve()
        save_accuracy_report(output, result, args.checkpoint, args.data_root,
                             args.split, weights.cpu().tolist())
        if (output/'metrics.csv').is_file():
            plot_loss_curve(output/'metrics.csv')
        print('Final report: {}'.format(output/'final_accuracy_report.txt'))


if __name__ == '__main__':
    main()
