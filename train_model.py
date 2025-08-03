import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

def build_model(input_shape=(8, 8, 12)):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='tanh')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

if __name__ == "__main__":
    with np.load("chess_dataset.npz") as data:
        X = data['X']
        y = data['y']

    model = build_model()
    model.summary()

    history = model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2)

    model.save("Cerebrus.h5")
    print("Модель сохранена в chess_eval_model.h5")