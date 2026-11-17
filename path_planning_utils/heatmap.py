"""Fixed-scale probability colors for the Tkinter map and its legend."""
import numpy as np


def probability_rgb(scores, opacity=0.55):
    """Blend a purple-to-yellow 0–1 probability scale over white.

    Use absolute probabilities, not per-map normalization: a probability of 0.5
    always has the same color, even when the model's scores have a narrow range.
    """
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Heatmap probabilities must be finite and between 0 and 1')
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('Heatmap opacity must be between 0 and 1')
    anchors = np.linspace(0, 1, 5)
    colors = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140),
                       (94, 201, 98), (253, 231, 37)], dtype=float)
    rgb = np.stack([np.interp(scores, anchors, colors[:, channel])
                    for channel in range(3)], axis=-1)
    return np.rint((1-opacity)*255+opacity*rgb).astype(np.uint8)
