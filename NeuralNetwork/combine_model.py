import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class TextDataGenerator:
    def __init__(self, directory, batch_size, num_classes):
        self.directory = directory
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.categories = sorted(os.listdir(directory))
        self.category_indices = {category: i for i, category in enumerate(self.categories)}

    def generate(self):
        while True:
            batch_features = []
            batch_labels = []

            for _ in range(self.batch_size):
                # 随机选择类别和文件
                category = np.random.choice(self.categories)
                category_dir = os.path.join(self.directory, category)
                file_choice = np.random.choice(os.listdir(category_dir))
                file_path = os.path.join(category_dir, file_choice)

                # 读取文件内容
                with open(file_path, 'r') as file:
                    vector = np.array([float(x) for x in file.read().split()])
                    batch_features.append(vector)

                # 添加标签
                batch_labels.append(self.category_indices[category])

            # 将列表转换为 Numpy 数组
            X = np.array(batch_features)
            y = to_categorical(batch_labels, num_classes=self.num_classes)

            yield X, y


class CombineModel:
    
    model: Sequential

    def __init__(self, input_dim=30, output_dim=10) -> None:
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.create_fc_model(input_dim, output_dim)
    
    
    # 定义全连接层模型
    def create_fc_model(self, input_dim=30, output_dim=10):  # 根据你的实际情况设置 output_dim
        adam = Adam
        self.model = Sequential()
        # self.model.add(Dense(units=60, input_dim=input_dim, activation='relu'))
        # self.model.add(Dense(units=64, activation='sigmoid'))
        # def full(shape, dtype=None):
        #     return np.full(shape, 1/30)
        # self.model.add(Dense(units=output_dim, input_dim=input_dim, activation='softmax', kernel_initializer=full))

        self.model.add(Conv1D(32, 3, padding="same", input_shape=(input_dim, 1), activation="relu"))
        self.model.add(Conv1D(64, 3, padding="same", activation="relu"))
        self.model.add(Conv1D(128, 3, padding="same", activation="relu"))
        self.model.add(Conv1D(256, 3, padding="same", activation="relu"))
        self.model.add(Conv1D(512, 3, padding="same", activation="relu"))
        self.model.add(Conv1D(1024, 3, padding="same", activation="relu"))
        self.model.add(Conv1D(2048, 3, padding="same", activation="relu"))
        self.model.add(Conv1D(4096, 3, padding="same", activation="relu"))
        self.model.add(Conv1D(8192, 3, padding="same", activation="relu"))
        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer=adam(learning_rate=1e-5), metrics=['acc'])
        print(self.model.summary())
        
    def train(self, input_path, batch_size=20, epoch=10, steps_per_epoch=50):
        data_gen = TextDataGenerator(input_path, batch_size, self.output_dim)
        history = self.model.fit(data_gen.generate(), steps_per_epoch=steps_per_epoch, epochs=epoch)

        # history = self.model.fit(np.expand_dims([6.148888170719146729e-02,8.227097243070602417e-02,2.801598608493804932e-02,7.997087389230728149e-02,1.317998319864273071e-01,1.262535452842712402e-01,6.778004020452499390e-02,6.190165877342224121e-02,2.619538605213165283e-01,9.856437891721725464e-02,1.684616655111312866e-01,8.227140456438064575e-02,1.129472777247428894e-01,2.601813524961471558e-02,3.145626932382583618e-02,4.898895621299743652e-01,1.065227668732404709e-02,1.902527362108230591e-02,4.129686951637268066e-02,1.798122003674507141e-02,4.714786540716886520e-03,6.701263919239863753e-05,4.951335489749908447e-01,4.115247167646884918e-03,1.278149778954684734e-03,3.635140135884284973e-03,5.510414950549602509e-03,8.266154909506440163e-04,3.956652712076902390e-03,4.807624518871307373e-01], axis=0), np.expand_dims([1,0,0,0,0,0,0,0,0,0], axis=0), epochs=epoch)
        
        loss = history.history["loss"]
        acc = history.history["acc"]

        epoch = range(epoch)
        plt.figure(1)
        plt.plot(epoch, acc, "bo", label="Training accuracy")
        plt.title("Training Accuracy")
        plt.legend()
        plt.savefig('./train_acc')


        plt.figure(2)
        plt.plot(epoch, loss, "bo", label="Training Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig('./train_loss')
        plt.show()


