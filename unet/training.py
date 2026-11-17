"""Shared train/evaluation loop with correct partial-batch and padding metrics."""
import torch
import torch.nn.functional as F


def run_epoch(model, loader, device, class_weights, optimizer=None,
              freeze_encoder=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_encoder:
        model.encoder.eval()  # preserve frozen BatchNorm state as well as weights
    total_loss, pixels, iou_sum, samples = 0.0, 0, 0.0, 0
    true_positive = false_positive = false_negative = true_negative = 0
    for inputs, labels, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if training and not freeze_encoder and hasattr(model, 'encoder'):
            model.encoder.train()
            # A final singleton 32x32 batch has one bottleneck activation.
            # Use accumulated BN statistics for that batch rather than dropping it.
            if inputs.shape[0]*(inputs.shape[-2]//32)*(inputs.shape[-1]//32) == 1:
                for module in model.encoder.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.eval()
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            # Normalize by actual valid pixels, including the last partial batch.
            loss_sum = F.cross_entropy(logits, labels, weight=class_weights,
                                       ignore_index=-100, reduction='sum')
            valid = labels != -100
            valid_count = int(valid.sum().item())
            loss = loss_sum/max(1, valid_count)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prediction = (logits.detach().argmax(1) == 1) & valid
        truth = (labels == 1) & valid
        true_positive += int((prediction & truth).sum().item())
        false_positive += int((prediction & ~truth & valid).sum().item())
        false_negative += int((~prediction & truth).sum().item())
        true_negative += int((~prediction & ~truth & valid).sum().item())
        intersection = (prediction & truth).flatten(1).sum(1).float()
        union = (prediction | truth).flatten(1).sum(1).float()
        iou = torch.where(union > 0, intersection/union.clamp_min(1), torch.ones_like(union))
        total_loss += float(loss_sum.detach().item())
        pixels += valid_count
        iou_sum += float(iou.sum().item())
        samples += inputs.shape[0]
    if not samples:
        raise ValueError('DataLoader yielded no samples')
    union_pixels = true_positive + false_positive + false_negative
    return {
        'loss': total_loss/pixels,
        'path_iou': iou_sum/samples,
        'samples': samples,
        'valid_pixels': pixels,
        'pixel_accuracy': (true_positive + true_negative)/pixels,
        'path_precision': true_positive/max(1, true_positive + false_positive),
        'path_recall': true_positive/max(1, true_positive + false_negative),
        'path_f1': 2*true_positive/max(1, 2*true_positive + false_positive + false_negative),
        'global_path_iou': true_positive/union_pixels if union_pixels else 1.0,
        'confusion_matrix': {
            'true_positive': true_positive, 'false_positive': false_positive,
            'false_negative': false_negative, 'true_negative': true_negative,
        },
    }

