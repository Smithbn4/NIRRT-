"""Headless learning curves and final segmentation reports for training runs."""
import csv
import json
from pathlib import Path


def plot_loss_curve(metrics_csv):
    """Plot all completed epochs and atomically replace the previous PNG."""
    metrics_csv = Path(metrics_csv)
    with metrics_csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [int(row['epoch']) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    output = metrics_csv.parent/'loss_curve.png'
    temporary = output.with_name('.loss_curve.tmp.png')
    try:
        for key, label in [('train_loss', 'Training'), ('val_loss', 'Validation')]:
            axis.plot(epochs, [float(row[key]) for row in rows],
                      label=label, linewidth=2, marker='o' if len(rows) < 20 else None)
        axis.set(title='ResNet50 / U-Net loss per epoch', xlabel='Completed epoch',
                 ylabel='Class-weighted cross-entropy loss')
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(epochs) == 1:
            axis.set_xticks(epochs)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(str(temporary), dpi=180)
        temporary.replace(output)
    finally:
        plt.close(figure)
    return output


def save_accuracy_report(output_dir, metrics, checkpoint, data_root, split,
                         class_weights, best_epoch=None, best_val_path_iou=None):
    """Save human-readable and machine-readable metrics for the evaluated model.

    Metrics are fractions in JSON and percentages in the text report. Path IoU
    retains the mean-per-query definition used to select training checkpoints.
    Precision, recall, F1, global IoU and pixel accuracy pool valid pixels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'checkpoint': str(Path(checkpoint).expanduser().resolve()),
        'data_root': str(Path(data_root).expanduser().resolve()),
        'split': split,
        'best_epoch': best_epoch,
        'best_validation_path_iou': best_val_path_iou,
        'class_weights': list(class_weights),
        'metrics': metrics,
        'notes': [
            'Segmentation against A* reference corridors; not planner success or path length.',
            'Targets use the default 3x3 path dilation. Padding pixels are excluded.',
            'path_iou is the mean per-query path IoU used for checkpoint selection.',
            'Other accuracy metrics pool valid pixels across the evaluated queries.',
            'Positive class = path; negative class = non-path (including obstacles).',
            'Pixel accuracy can be high when path pixels are rare; also inspect IoU and F1.',
            'Precision, recall and F1 are 0 when their denominator is 0; empty-union IoU is 1.',
            'JSON accuracy metrics are fractions from 0 to 1, not percentages.',
        ],
    }
    lines = [
        'FINAL U-NET SEGMENTATION ACCURACY REPORT',
        'Checkpoint: {}'.format(report['checkpoint']),
        'Dataset: {}'.format(report['data_root']),
        'Evaluation split: {}'.format(split),
        'Evaluated queries: {}'.format(metrics['samples']),
        'Valid pixels: {}'.format(metrics['valid_pixels']),
    ]
    if best_epoch is not None:
        lines.append('Selected epoch: {}'.format(best_epoch))
    if best_val_path_iou is not None:
        lines.append('Selection validation path IoU: {:.2%}'.format(best_val_path_iou))
    lines += ['', 'Weighted cross-entropy loss: {:.6f}'.format(metrics['loss'])]
    for key, label in [
        ('pixel_accuracy', 'Pixel accuracy'), ('path_iou', 'Mean per-query path IoU'),
        ('global_path_iou', 'Global path IoU'), ('path_precision', 'Path precision'),
        ('path_recall', 'Path recall'), ('path_f1', 'Path F1 / Dice'),
    ]:
        lines.append('{}: {:.2%}'.format(label, metrics[key]))
    lines += ['', 'Confusion matrix (positive = path):']
    lines += ['  {}: {}'.format(key, value)
              for key, value in metrics['confusion_matrix'].items()]
    lines += ['', 'Interpretation:'] + ['- '+note for note in report['notes']]

    for name, contents in [
        ('final_accuracy_report.json', json.dumps(report, indent=2, allow_nan=False)+'\n'),
        ('final_accuracy_report.txt', '\n'.join(lines)+'\n'),
    ]:
        output = output_dir/name
        temporary = output.with_name('.'+name+'.tmp')
        temporary.write_text(contents)
        temporary.replace(output)
    return report
