# 目前进度
- 基础模型搭建：使用了CNN对Mel频谱图的识别，对正常歌曲识别率不高（40%左右？比较偏好reggae分类，但是对于原来的训练数据集的预测有问题（指随机采样的预测））
- 同时，我们将分类进行放大，但是保留原有的十个分类进行训练和预测，只是在输出时只输出四个分类：pop(`pop,blues,country,disco,hiphop,reggae`),classical,rock(`rock,metal`),jazz
- 数据的预处理：将歌曲转成频谱图，然后随机采样截取5张(128, 128)大小的图片（大约3秒）作为训练数据集（这个是否需要修改？因为可能会有重复的部分出现），对于测试（验证）数据集，在相同的歌曲片段（30s）的基础上，随机截取一张(128, 128)的图像
- 模型的训练预测保存载入等基本操作（对于CNN）完成，可以照葫芦画瓢来写

# 现在的任务
- [ ] 考虑这些，看看能不能使用MLP全连接直接训练一个新的“弱”分类器，用boosting联合投票
    ```python
    # 1. 音高和旋律特征
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # 2. 节奏和节拍
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # 3. 和声和和弦结构
    harmonic = librosa.effects.harmonic(y)
    harmonic_features = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    
    # 4. 音色和乐器使用
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    # 5. 动态范围和强度
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    
    # 6. 时间结构和形式
    temporal_onset = librosa.onset.onset_strength(y=y, sr=sr)
    
    # 7. 时域特征
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    ```
    详细说明参见`./NerualNetwork/classifier.ipynb`最后一个单元格
- [ ] 根据新的数据点更新模型，好像使用`model.fit`直接训练？不确定

    具体怎么做要试一下
- [ ] GUI 界面（待定）

# 分工

来认领任务！！！别全让我一个人来写！！！

你们可以尝试运行`./NeuralNetwork/data-preprocess.py`生成图片的训练数据，因为太多我就不上传了
