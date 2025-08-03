import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt

INPUT_SHAPE = (8, 8, 12)
BATCH_SIZE = 64
EPOCHS = 30
DATA_FILE = "dataset/chess_dataset.npz"
MODEL_FILE = "Cerebrus.h5"


def residual_block(x, filters, kernel_size, change_filters=False):
    """
    Implements a basic residual block with optional filter dimension change.
    """
    shortcut = x
    if change_filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    return layers.Activation('relu')(x)


def build_model():
    """
    Constructs the Cerebrus CNN model with residual connections.
    """
    inp = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)

    for _ in range(4):
        x = residual_block(x, 64, 3)

    x = residual_block(x, 128, 3, change_filters=True)
    for _ in range(2):
        x = residual_block(x, 128, 3)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='tanh')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, mae, label='Train MAE')
    plt.plot(epochs_range, val_mae, label='Validation MAE')
    plt.title('Mean Absolute Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = np.load(DATA_FILE)
    X, y = data['X'], data['y']

    model = build_model()
    model.summary()

    checkpoint_cb = callbacks.ModelCheckpoint(
        f"best_{MODEL_FILE}", save_best_only=True,
        monitor='val_loss', mode='min'
    )
    earlystop_cb = callbacks.EarlyStopping(
        patience=5, restore_best_weights=True,
        monitor='val_loss'
    )

    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=2
    )

    plot_history(history)

    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
