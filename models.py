import keras
import numpy as np
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu


def widths(scale):
    return [max(4, int(64 * scale)), max(4, int(32 * scale)), max(4, int(32 * scale))]


def build_model(bits, scale, n_in=16, n_out=5):
    """bits=None -> float baseline. Otherwise QKeras with `bits` for weights and activations."""
    model = keras.Sequential(name="jet_tagger")
    model.add(keras.Input(shape=(n_in,), name="jet_features"))
    for i, n in enumerate(widths(scale), start=1):
        if bits is None:
            model.add(keras.layers.Dense(n, name=f"fc{i}", kernel_initializer="lecun_uniform"))
            model.add(keras.layers.Activation("relu", name=f"relu{i}"))
        else:
            q = quantized_bits(bits, 0, alpha=1)
            model.add(QDense(n, name=f"fc{i}", kernel_quantizer=q, bias_quantizer=q,
                             kernel_initializer="lecun_uniform"))
            model.add(QActivation(quantized_relu(bits), name=f"relu{i}"))
    if bits is None:
        model.add(keras.layers.Dense(n_out, name="output", kernel_initializer="lecun_uniform"))
    else:
        q = quantized_bits(bits, 0, alpha=1)
        model.add(QDense(n_out, name="output", kernel_quantizer=q, bias_quantizer=q,
                         kernel_initializer="lecun_uniform"))
    model.add(keras.layers.Activation("softmax", name="softmax"))
    return model


def make_masks(model, sparsity):
    """Zero the smallest-magnitude `sparsity` fraction of each dense kernel. Returns (layer, mask) pairs."""
    masks = []
    for layer in model.layers:
        if layer.name.startswith(("fc", "output")):
            w = layer.get_weights()
            threshold = np.quantile(np.abs(w[0]), sparsity)
            mask = (np.abs(w[0]) > threshold).astype(w[0].dtype)
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