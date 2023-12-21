import os

import keras
import matplotlib.pyplot as plt
import numpy as np


class CNN:
    cnn: keras.models.Sequential

    def __init__(self, input_shape: tuple):
        conv = keras.layers.Conv2D
        dense = keras.layers.Dense
        max_pool = keras.layers.MaxPool2D
        avg_pool = keras.layers.AvgPool2D
        dropout = keras.layers.Dropout
        flatten = keras.layers.Flatten

        adam = keras.optimizers.Adam

        imageGenerator = keras.preprocessing.image.ImageDataGenerator

        self.cnn = keras.models.Sequential()
        self.cnn.add(conv(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(64, (3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(128, (3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.cnn.add(max_pool(pool_size=(2, 2)))
        self.cnn.add(conv(256, (3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.cnn.add(flatten())
        self.cnn.add(dense(1024, activation="relu"))
        self.cnn.add(dense(512, activation="relu"))
        self.cnn.add(dropout(0.5))
        self.cnn.add(dense(10, activation="softmax"))

        self.cnn.compile(loss="categorical_crossentropy", optimizer=adam(learning_rate=1e-4), metrics=["acc"])

        self.train_datagen = imageGenerator(rescale=1. / 255)
        self.test_datagen = imageGenerator(rescale=1. / 255)

        print(self.cnn.summary())

    def save_model(self, path: str) -> None:
        """
        save model to path
        :param path: the parent dir of model
        """
        os.makedirs(path, exist_ok=True)
        self.cnn.save(os.path.join(path, "model.h5"))

    def load_model(self, path: str) -> None:
        """
        load the model form path
        :param path: model path(ends with ".h5")
        """
        self.cnn = keras.models.load_model(path)

    def train(self, train_path: str, test_path: str | None = None, epoch: int = 30) -> None:
        train_generator = self.train_datagen.flow_from_directory(train_path,
                                                                 color_mode="gray",
                                                                 target_size=(128, 128),
                                                                 batch_size=20,
                                                                 class_mode="one-hot")
        # test_generator = self.test_datagen.flow_from_directory(test_path,
        #                                                        color_mode="gray",
        #                                                        target_size=(128, 128),
        #                                                        batch_size=20,
        #                                                        class_mode="one-hot")

        history = self.cnn.fit_generator(train_generator,
                                         steps_per_epoch=250,
                                         epochs=epoch,
                                         verbose=2)

        acc = history["acc"]
        loss = history["loss"]

        plt.plot(epoch, acc, "bo", label="Training accuracy")
        plt.plot(epoch, loss, "b", label="Training Loss")
        plt.title("Training Accuracy and Loss")
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.cnn.predict(X)

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        self.cnn.fit(X, y, batch_size=32, verbose=1, epochs=1)
