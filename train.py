import keras
import numpy as np
from keras import ops

from models import KeepPruned, build_model, make_masks

LOSS = "sparse_categorical_crossentropy"
MIN_DELTA = 5e-4  # val_loss must improve by at least this much to count, so training stops once gains are negligible


def soft_cross_entropy(y_true, y_pred):
    """Custom loss (v2), replacing Keras's built-in: cross-entropy against a whole target distribution
    instead of one integer label, so the targets can carry the teacher model's soft answers."""
    return -ops.sum(y_true * ops.log(ops.clip(y_pred, 1e-7, 1.0)), axis=-1)


def soft_targets(y, teacher_probs, alpha=0.5, n_classes=5):
    """Distillation targets (v2): alpha of the true label plus (1 - alpha) of the teacher's probabilities.
    The true class always keeps at least alpha, so the label itself never flips."""
    return (alpha * np.eye(n_classes, dtype=np.float32)[y] + (1 - alpha) * teacher_probs).astype(np.float32)


def train(cfg, X_tr, y_tr, X_val=None, y_val=None, targets=None, seed=0):
    """targets=None -> hard labels and the built-in loss (v1). Otherwise soft targets and soft_cross_entropy."""
    keras.utils.set_random_seed(seed)
    model = build_model(cfg["bits"], cfg["scale"], n_in=X_tr.shape[1])
    if targets is None:
        loss, metric, y_fit = LOSS, "accuracy", y_tr
        val = None if X_val is None else (X_val, y_val)
    else:
        loss, metric, y_fit = soft_cross_entropy, "categorical_accuracy", targets
        val = (X_val, np.eye(targets.shape[1], dtype=np.float32)[y_val])  # judge on the true labels
    split = dict(validation_split=0.2) if val is None else dict(validation_data=val)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=[metric])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=MIN_DELTA,
                                         restore_best_weights=True)
    model.fit(X_tr, y_fit, batch_size=1024, epochs=100, callbacks=[stop], verbose=2, **split)

    if cfg["sparsity"] > 0:
        masks = make_masks(model, cfg["sparsity"])
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=loss, metrics=[metric])
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, min_delta=MIN_DELTA,
                                             restore_best_weights=True)
        model.fit(X_tr, y_fit, batch_size=1024, epochs=15, callbacks=[KeepPruned(masks), stop], verbose=2, **split)
        for layer, mask in masks:
            w = layer.get_weights()
            w[0] = w[0] * mask
            layer.set_weights(w)
    return model
