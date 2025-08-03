import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

INPUT_SHAPE = (8, 8, 12)
BATCH_SIZE = 64
EPOCHS = 30
DATA_FILE = "dataset/chess_dataset.npz"
MODEL_FILE = "Cerebrus.h5"

def residual_block(x, filters, kernel_size):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    return layers.Activation('relu')(x)

def build_model():
    inp = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)

    for _ in range(4):
        x = residual_block(x, 64, 3)
    for _ in range(3):
        x = residual_block(x, 128, 3)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='tanh')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

if __name__ == "__main__":
    data = np.load(DATA_FILE)
    X, y = data['X'], data['y']

    model = build_model()
    model.summary()

    cp = callbacks.ModelCheckpoint("best_"+MODEL_FILE,
                                   save_best_only=True,
                                   monitor='val_loss',
                                   mode='min')
    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[cp, es],
        verbose=2
    )

    model.save(MODEL_FILE)
    print(f"Модель сохранена в {MODEL_FILE}")
