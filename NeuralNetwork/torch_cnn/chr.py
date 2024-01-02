from NeuralNetwork.torch_cnn.base_model import CNN

class CHR_CNN(CNN):
    def __init__(self):
        super().__init__(256 * (12 // 2 ** 3) * (1293 // 2 ** 3))