import os
import random

import keras
import matplotlib.pyplot as plt
import numpy as np


class CNN:
    cnn: keras.models.Sequential
    train_generator: keras.preprocessing.image.DirectoryIterator
    test_generator: keras.preprocessing.image.DirectoryIterator
    train_datagen: keras.preprocessing.image.ImageDataGenerator
    test_datagen: keras.preprocessing.image.ImageDataGenerator

    def __init__(self, input_shape: tuple):
        conv = keras.layers.Conv2D
        dense = keras.layers.Dense
        max_pool = keras.layers.MaxPool2D
        avg_pool = keras.layers.AvgPool2D
        dropout = keras.layers.Dropout
        batch_norm = keras.layers.BatchNormalization
        flatten = keras.layers.Flatten

        adam = keras.optimizers.Adam

        imageGenerator = keras.preprocessing.image.ImageDataGenerator

        self.cnn = keras.models.Sequential()
        self.cnn.add(conv(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(64, (3, 3), padding="same", activation="relu"))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(128, (3, 3), padding="same", activation="relu"))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(256, (3, 3), padding="same", activation="relu"))
        self.cnn.add(batch_norm())
        self.cnn.add(dropout(0.25))
        self.cnn.add(flatten())
        self.cnn.add(dense(1024, activation="relu"))
        self.cnn.add(dense(512, activation="sigmoid"))
        self.cnn.add(dropout(0.5))
        self.cnn.add(dense(10, activation="softmax"))

        self.cnn.compile(loss="categorical_crossentropy", optimizer=adam(learning_rate=1e-4), metrics=["acc"])

        self.train_datagen = imageGenerator(rescale=1. / 255)
        self.test_datagen = imageGenerator(rescale=1. / 255)

        print(self.cnn.summary())

    def save_model(self, path: str = "./model/model.h5") -> None:
        """
        save model to path
        :param path: the path of model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.cnn.save(path)

    def load_model(self, path: str = "./model/model.h5") -> None:
        """
        load the model form path
        :param path: model path(ends with ".h5")
        """
        self.cnn = keras.models.load_model(path)

    def train(self, train_path: str, test_path: str | None = None, epoch: int = 10) -> None:
        self.train_generator = self.train_datagen.flow_from_directory(train_path,
                                                                      color_mode="grayscale",
                                                                      target_size=(128, 128),
                                                                      batch_size=20,
                                                                      class_mode="categorical")
        self.test_generator = self.test_datagen.flow_from_directory(test_path,
                                                                    color_mode="grayscale",
                                                                    target_size=(128, 128),
                                                                    batch_size=20,
                                                                    class_mode="categorical")

        history = self.cnn.fit(self.train_generator,
                               steps_per_epoch=250,
                               epochs=epoch,
                               validation_data=self.test_generator,
                               validation_steps=50,
                               verbose=2)

        acc = history.history["acc"]
        val_acc = history.history["val_acc"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epoch = range(epoch)

        plt.plot(epoch, acc, "bo", label="Training accuracy")
        plt.plot(epoch, val_acc, "b", label="Training accuracy")
        plt.title("Training and Testing Accuracy")

        plt.figure()

        plt.plot(epoch, loss, "bo", label="Training Loss")
        plt.plot(epoch, val_loss, "b", label="Training Loss")
        plt.title("Training and Testing Loss")
        plt.legend()

        plt.show()

    def predict(self, X: np.ndarray, duration: float) -> str:
        predictions = []
        for i in range(round(duration / 10)):
            start_pos = random.randint(0, X.shape[1] - 128 - 1)
            pic = X[:, start_pos:start_pos + 128].reshape((1, 128, 128, 1))
            prediction = self.cnn.predict(pic)
            if np.max(prediction) > 0.3:
                predictions.append(prediction)

        predictions = np.mean(predictions, axis=0)
        print(predictions)
        if np.max(predictions) <= 0.4:
            return "Unknown"
        class_index = np.argmax(predictions, axis=1)[0]
        return ["pop", "classical", "pop", "pop", "pop", "jazz", "rock", "pop", "pop", "rock"][class_index]

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
