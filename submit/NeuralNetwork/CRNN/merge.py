import numpy as np
from keras.layers import Dense
from keras.models import Sequential


# ??? what's this

class CRNN_MERGE(CRNN):
    def __init__(self, input_shape: tuple):
        super().__init__(input_shape)

    def init(self, input_shape: tuple):
        model = Sequential()
        model.add(Dense(units=128, input_dim=input_dim, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, train_path: str, test_path: str | None = None, epoch: int = 10) -> None:
        self._train_core(train_path, test_path, epoch, (12, 1293), 20, 50)

    def read_array_from_txt(input_path):
        try:
            array_data = np.loadtxt(input_path)
            return array_data.reshape(1, -1)  # 将数据 reshape 成 (1, 30)
        except Exception as e:
            print(f"Error reading array from {input_path}: {e}")
            return None
    
    def predict(self, X: np.ndarray, duration: float) -> [str, np.ndarray]:
        predictions = []
        for i in range(round(duration / 30)):
            start_pos = random.randint(0, X.shape[1] - 1293)
            pic = X[:, start_pos: start_pos + 1293].reshape((1, 12, 1293, 1))
            prediction = self.cnn.predict(pic)
            predictions.append(prediction)

        predictions = np.mean(predictions, axis=0)
        class_index = np.argmax(predictions, axis=1)[0]
        return ["pop", "classical", "pop", "pop", "pop", "jazz", "rock", "pop", "pop", "rock"][class_index], predictions

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

