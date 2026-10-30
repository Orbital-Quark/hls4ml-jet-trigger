import keras

from models import KeepPruned, build_model, make_masks

LOSS = "sparse_categorical_crossentropy"


def train(cfg, X_tr, y_tr, seed=0):
    keras.utils.set_random_seed(seed)
    model = build_model(cfg["bits"], cfg["scale"])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=LOSS, metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, batch_size=1024, epochs=100, validation_split=0.2, callbacks=[stop], verbose=2)

    if cfg["sparsity"] > 0:
        masks = make_masks(model, cfg["sparsity"])
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=LOSS, metrics=["accuracy"])
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr, batch_size=1024, epochs=15, validation_split=0.2,
                  callbacks=[KeepPruned(masks), stop], verbose=2)
        for layer, mask in masks:
            w = layer.get_weights()
            w[0] = w[0] * mask
            layer.set_weights(w)
    return model