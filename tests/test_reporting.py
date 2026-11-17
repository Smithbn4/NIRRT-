"""Check accuracy aggregation with padding and a partial final batch."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from unet.reporting import plot_loss_curve, save_accuracy_report
from unet.training import run_epoch


class PredictionModel(torch.nn.Module):
    def forward(self, inputs):
        return torch.cat([-inputs, inputs], dim=1)


class ReportingTests(unittest.TestCase):
    def test_confusion_metrics_ignore_padding_and_weight_actual_pixels(self):
        labels = torch.tensor([
            [[1, 0, -100], [1, 0, 0]],
            [[1, 1, -100], [0, 0, -100]],
            [[0, 0, -100], [1, 0, -100]],
        ])
        predicted = torch.tensor([
            [[1, 1, 1], [0, 0, 0]],
            [[1, 1, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1]],
        ])
        inputs = (2*predicted.float()-1).unsqueeze(1)
        loader = DataLoader(TensorDataset(inputs, labels, torch.arange(3)), batch_size=2)
        metrics = run_epoch(PredictionModel(), loader, torch.device('cpu'), torch.ones(2))
        self.assertEqual(metrics['samples'], 3)
        self.assertEqual(metrics['valid_pixels'], 13)
        self.assertEqual(metrics['confusion_matrix'], {
            'true_positive': 4, 'false_positive': 2, 'false_negative': 1, 'true_negative': 6})
        for key, expected in [('pixel_accuracy', 10/13), ('path_precision', 4/6),
                              ('path_recall', 4/5), ('path_f1', 8/11),
                              ('global_path_iou', 4/7), ('path_iou', 2/3)]:
            self.assertAlmostEqual(metrics[key], expected, places=6)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_accuracy_report(root, metrics, root/'best_unet.pt', root/'data',
                                 'test', [1.0, 2.0], best_epoch=7, best_val_path_iou=0.5)
            report = json.loads((root/'final_accuracy_report.json').read_text())
            self.assertEqual(report['best_epoch'], 7)
            self.assertEqual(report['metrics']['valid_pixels'], 13)
            self.assertIn('Pixel accuracy: 76.92%', (root/'final_accuracy_report.txt').read_text())

    def test_empty_positive_class_has_explicit_finite_metrics(self):
        inputs = -torch.ones(1, 1, 2, 2)
        labels = torch.zeros(1, 2, 2, dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, labels, torch.arange(1)))
        metrics = run_epoch(PredictionModel(), loader, torch.device('cpu'), torch.ones(2))
        self.assertEqual(metrics['pixel_accuracy'], 1)
        self.assertEqual(metrics['path_iou'], 1)
        self.assertEqual(metrics['global_path_iou'], 1)
        self.assertEqual(metrics['path_precision'], 0)
        self.assertEqual(metrics['path_recall'], 0)
        self.assertEqual(metrics['path_f1'], 0)

    def test_curve_is_refreshed_from_completed_csv_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory)/'metrics.csv'
            with csv_path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            self.assertIsNone(plot_loss_curve(csv_path))
            with csv_path.open('a', newline='') as stream:
                csv.writer(stream).writerow([1, 0.5, 0.6])
            output = plot_loss_curve(csv_path)
            first = output.read_bytes()
            self.assertTrue(first.startswith(b'\x89PNG\r\n\x1a\n'))
            with csv_path.open('a', newline='') as stream:
                csv.writer(stream).writerow([2, 0.25, 0.4])
            plot_loss_curve(csv_path)
            self.assertNotEqual(first, output.read_bytes())
            self.assertFalse((Path(directory)/'.loss_curve.tmp.png').exists())


if __name__ == '__main__':
    unittest.main()
