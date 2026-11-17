"""Train or fine-tune the 2D ResNet50/U-Net path segmenter."""
import argparse
import csv
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from project_config import ROOT, DEFAULT_DATA, DEFAULT_CHECKPOINT, select_device, seed_everything, load_model_weights
from unet.image_dataset import ImageDataset
from unet.models.unet_model import UNet
from unet.training import run_epoch
from unet.reporting import plot_loss_curve, save_accuracy_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/training')
    parser.add_argument('--init-checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--from-scratch', action='store_true', help='Do not load a trained U-Net checkpoint; train all encoder parameters')
    parser.add_argument('--imagenet', action='store_true', help='With --from-scratch, initialize the encoder from ImageNet (may download weights)')
    parser.add_argument('--train-encoder', action='store_true', help='Fine-tune encoder parameters and BatchNorm state too')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-val-samples', type=int)
    parser.add_argument('--max-test-samples', type=int, help='Limit final test evaluation (for pipeline checks only)')
    parser.add_argument('--skip-test-evaluation', action='store_true', help='Skip the final test report, e.g. for datasets without a test split')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.workers < 0:
        parser.error('epochs, batch size and learning rate must be positive; workers must be nonnegative')
    if args.imagenet and not args.from_scratch:
        parser.error('--imagenet requires --from-scratch')
    if args.from_scratch and args.batch_size < 2:
        parser.error('Use batch size >= 2 when training encoder BatchNorm on small images')
    device = select_device(args.device, args.threads)
    seed_everything(args.seed)
    args.output = args.output.expanduser().resolve()
    # A new run prevents accidental replacement of supplied or previous weights.
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {
        'train': ImageDataset(args.data_root, 'train', augment=True, limit=args.max_train_samples),
        'val': ImageDataset(args.data_root, 'val', limit=args.max_val_samples),
    }
    # Check test data before the first epoch rather than after a long training run.
    test_dataset = None if args.skip_test_evaluation else ImageDataset(
        args.data_root, 'test', limit=args.max_test_samples)
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=name == 'train', num_workers=args.workers)
               for name, dataset in datasets.items()}
    frozen = not (args.train_encoder or args.from_scratch)
    model = UNet(freeze_original_weights=frozen, pretrained=args.imagenet).to(device)
    if not args.from_scratch:
        load_model_weights(model, args.init_checkpoint, device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    weights = torch.tensor(datasets['train'].class_weights(), device=device)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(device=str(device), class_weights=weights.cpu().tolist(), freeze_encoder=frozen)
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2))
    best = -1.0
    best_epoch = None
    with (args.output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_path_iou', 'val_path_iou', 'seconds',
                         'train_pixel_accuracy', 'val_pixel_accuracy'])
        for epoch in range(1, args.epochs+1):
            started = time.perf_counter()
            train = run_epoch(model, loaders['train'], device, weights, optimizer, frozen)
            val = run_epoch(model, loaders['val'], device, weights)
            writer.writerow([epoch, train['loss'], val['loss'], train['path_iou'], val['path_iou'],
                             time.perf_counter()-started, train['pixel_accuracy'], val['pixel_accuracy']])
            stream.flush()
            if val['path_iou'] > best:
                best = val['path_iou']
                best_epoch = epoch
                # Keep the last complete checkpoint if saving is interrupted.
                temporary = args.output/'.best_unet.tmp.pt'
                torch.save(model.state_dict(), str(temporary))
                temporary.replace(args.output/'best_unet.pt')
            plot_loss_curve(args.output/'metrics.csv')
            print('Epoch {}/{}: train loss {:.4f}, val loss {:.4f}, val path IoU {:.4f}'.format(
                epoch, args.epochs, train['loss'], val['loss'], val['path_iou']), flush=True)
    print('Best checkpoint: {}'.format(args.output/'best_unet.pt'))
    print('Loss curve: {}'.format(args.output/'loss_curve.png'))
    if test_dataset is not None:
        checkpoint = args.output/'best_unet.pt'
        load_model_weights(model, checkpoint, device)
        print('Evaluating the best checkpoint on {} test queries...'.format(len(test_dataset)), flush=True)
        test = run_epoch(model, DataLoader(test_dataset, batch_size=args.batch_size,
                                         num_workers=args.workers), device, weights)
        save_accuracy_report(args.output, test, checkpoint, args.data_root, 'test',
                             weights.cpu().tolist(), best_epoch, best)
        print('Test pixel accuracy: {:.2%}; path IoU: {:.2%}; path F1: {:.2%}'.format(
            test['pixel_accuracy'], test['path_iou'], test['path_f1']))
        print('Final report: {}'.format(args.output/'final_accuracy_report.txt'))


if __name__ == '__main__':
    main()

