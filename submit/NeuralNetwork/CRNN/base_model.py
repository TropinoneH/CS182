import os

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class CRNN:
    cnn: keras.models.Sequential
    train_generator: keras.preprocessing.image.DirectoryIterator
    test_generator: keras.preprocessing.image.DirectoryIterator
    train_datagen: keras.preprocessing.image.ImageDataGenerator
    test_datagen: keras.preprocessing.image.ImageDataGenerator

    def __init__(self, input_shape: tuple):
        adam = keras.optimizers.Adam
        self.init(input_shape)

        self.cnn.compile(
            loss="categorical_crossentropy",
            optimizer=adam(learning_rate=1e-4),
            metrics=["acc"],
        )

        print(self.cnn.summary())


    def init(self, input_shape: tuple):
        raise NotImplementedError("the init function doesn't implemented: ", self.__class__.__name__)

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

    def _train_core(self, train_path: str, test_path: str | None, epoch: int, target_size: tuple, batch_size: int,
                    steps_per_epoch: int):
        self.train_generator = self.train_datagen.flow_from_directory(train_path,
                                                                      color_mode="grayscale",
                                                                      target_size=target_size,
                                                                      batch_size=batch_size,
                                                                      class_mode="categorical")
        if test_path is not None:
            self.test_generator = self.test_datagen.flow_from_directory(test_path,
                                                                        color_mode="grayscale",
                                                                        target_size=target_size,
                                                                        batch_size=batch_size,
                                                                        class_mode="categorical")

            history = self.cnn.fit(self.train_generator,
                                   steps_per_epoch=steps_per_epoch,
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
        else:
            history = self.cnn.fit(self.train_generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=epoch,
                                   verbose=2)

            acc = history.history["acc"]
            loss = history.history["loss"]

            epoch = range(epoch)

            plt.plot(epoch, acc, "bo", label="Training accuracy")
            plt.title("Training and Testing Accuracy")

            plt.figure()

            plt.plot(epoch, loss, "bo", label="Training Loss")
            plt.title("Training and Testing Loss")
            plt.legend()

            plt.show()

    def train(self, train_path: str, test_path: str | None = None, epoch: int = 10) -> None:
        raise NotImplementedError("the train function doesn't implemented: ", self.__class__.__name__)

    def predict(self, X: np.ndarray, duration: float) -> [str, np.ndarray]:
        raise NotImplementedError("the predict function doesn't implemented: ", self.__class__.__name__)

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
