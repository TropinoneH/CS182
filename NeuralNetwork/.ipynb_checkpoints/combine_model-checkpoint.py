import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class CombineModel:
    
    def read_array_from_txt(input_path):
            try:
                array_data = np.loadtxt(input_path)
                return array_data.reshape(1, -1)  # 将数据 reshape 成 (1, 30)
            except Exception as e:
                print(f"Error reading array from {input_path}: {e}")
                return None
    
    # 定义全连接层模型
    def create_fc_model(input_dim=30, output_dim=10):  # 根据你的实际情况设置 output_dim
        model = Sequential()
        model.add(Dense(units=128, input_dim=input_dim, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

input_data_path = "/home/lumiani/Projects/IML/CS182-FinalProject/dataset/merge_format/blues/blues.00000.txt"
model_output_dim = 10

# 读取数组数据
array_data = read_array_from_txt(input_data_path)
print(array_data)

# 如果成功读取数据，则使用全连接层模型进行训练
if array_data is not None:
    # 创建全连接层模型
    model = create_fc_model(input_dim=array_data.shape[1], output_dim=model_output_dim)

    target_data = np.random.rand(1, model_output_dim)

    # 训练模型
    model.fit(array_data, target_data, epochs=10, batch_size=1)
else:
    print("Error: Unable to proceed with training due to missing array data.")
