from NeuralNetwork.torch_cnn.base_model import CNN

class MEL_CNN(CNN):
    def __init__(self):
        super(MEL_CNN, self).__init__(256 * 16 * 16)
