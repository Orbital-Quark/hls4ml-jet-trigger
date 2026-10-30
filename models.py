import keras
import numpy as np
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu


def widths(scale):
    return [max(4, int(64 * scale)), max(4, int(32 * scale)), max(4, int(32 * scale))]


def build_model(bits, scale, n_in=16, n_out=5):
    """bits=None -> float baseline. An int -> QKeras with `bits` for weights and activations.
    A 4-tuple -> bits per dense layer (fc1, fc2, fc3, output), e.g. (8, 4, 4, 8) (v2 per-layer bit widths)."""
    per_layer = list(bits) if isinstance(bits, (tuple, list)) else [bits] * 4
    model = keras.Sequential(name="jet_tagger")
    model.add(keras.Input(shape=(n_in,), name="jet_features"))
    for i, (n, b) in enumerate(zip(widths(scale), per_layer), start=1):
        if b is None:
            model.add(keras.layers.Dense(n, name=f"fc{i}", kernel_initializer="lecun_uniform"))
            model.add(keras.layers.Activation("relu", name=f"relu{i}"))
        else:
            q = quantized_bits(b, 0, alpha=1)
            model.add(QDense(n, name=f"fc{i}", kernel_quantizer=q, bias_quantizer=q,
                             kernel_initializer="lecun_uniform"))
            model.add(QActivation(quantized_relu(b), name=f"relu{i}"))
    b = per_layer[3]
    if b is None:
        model.add(keras.layers.Dense(n_out, name="output", kernel_initializer="lecun_uniform"))
    else:
        q = quantized_bits(b, 0, alpha=1)
        model.add(QDense(n_out, name="output", kernel_quantizer=q, bias_quantizer=q,
                         kernel_initializer="lecun_uniform"))
    model.add(keras.layers.Activation("softmax", name="softmax"))
    return model


def make_masks(model, sparsity):
    """Zero the smallest-magnitude `sparsity` fraction of each dense kernel. Returns (layer, mask) pairs.
    Keeps exactly the largest (1 - sparsity) by rank: QKeras clips weights at +-1, so many can tie at the
    same magnitude, and a `> threshold` cut would then delete all of them (it killed whole output layers)."""
    masks = []
    for layer in model.layers:
        if layer.name.startswith(("fc", "output")):
            w = layer.get_weights()
            mag = np.abs(w[0]).ravel()
            keep = np.argsort(-mag, kind="stable")[:int(round(mag.size * (1 - sparsity)))]
            mask = np.zeros(mag.size, dtype=w[0].dtype)
            mask[keep] = 1
            mask = mask.reshape(w[0].shape)
            w[0] = w[0] * mask
            layer.set_weights(w)
            masks.append((layer, mask))
    return masks


class KeepPruned(keras.callbacks.Callback):
    """Re-applies the masks after every batch so pruned weights stay at zero."""

    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def on_train_batch_end(self, batch, logs=None):
        for layer, mask in self.masks:
            w = layer.get_weights()
            w[0] = w[0] * mask
            layer.set_weights(w)