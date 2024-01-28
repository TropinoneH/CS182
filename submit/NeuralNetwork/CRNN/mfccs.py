import random

import keras
import numpy as np

from NeuralNetwork.CRNN.base_model import CRNN


class CRNN_MFCC(CRNN):
    def __init__(self, input_shape: tuple):
        super().__init__(input_shape)

    def init(self, input_shape: tuple):
        conv = keras.layers.Conv2D
        dense = keras.layers.Dense
        max_pool = keras.layers.MaxPool2D
        avg_pool = keras.layers.AvgPool2D
        dropout = keras.layers.Dropout
        batch_norm = keras.layers.LayerNormalization
        flatten = keras.layers.Flatten
        time_distributed = keras.layers.TimeDistributed
        lstm = keras.layers.LSTM

        imageGenerator = keras.preprocessing.image.ImageDataGenerator

        self.cnn = keras.models.Sequential()

        self.cnn.add(conv(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(64, (3, 3), padding="same", activation="relu"))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(128, (3, 3), padding="same", activation="relu"))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        # self.cnn.add(conv(256, (3, 3), padding="same", activation="relu"))
        self.cnn.add(batch_norm())
        self.cnn.add(time_distributed(flatten()))
        self.cnn.add(lstm(256, return_sequences=False))
        self.cnn.add(dense(1024, activation="relu"))
        self.cnn.add(dense(512, activation="sigmoid"))
        self.cnn.add(dropout(0.5))
        self.cnn.add(dense(10, activation="softmax"))

        self.train_datagen = imageGenerator(featurewise_center=True,featurewise_std_normalization=True,zca_whitening=True,rescale=1.0 / 255)
        self.test_datagen = imageGenerator(rescale=1.0 / 255)

    def train(self, train_path: str, test_path: str | None = None, epoch: int = 10) -> None:
        self._train_core(train_path, test_path, epoch, (1293,20), 20, 50)

    def predict(self, X: np.ndarray, duration: float) -> [str, np.ndarray]:
        predictions = []
        for i in range(round(duration / 30)):
            start_pos = random.randint(0, X.shape[1] - 1293)
            pic = X[:, start_pos: start_pos + 1293].reshape((1, 1293, 20,  1))
            prediction = self.cnn.predict(pic)
            if np.max(prediction) > 0.3:
                predictions.append(prediction)

        predictions = np.mean(predictions, axis=0)
        class_index = np.argmax(predictions, axis=1)[0]
        return ["pop", "classical", "pop", "pop", "pop", "jazz", "rock", "pop", "pop", "rock"][class_index], predictions

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
