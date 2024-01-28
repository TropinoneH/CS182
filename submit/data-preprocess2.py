import os
import random

from NeuralNetwork.CRNN import CRNN_MEL, CRNN_MFCC
from NeuralNetwork.CNN import CNN_CHR

model_mel = CRNN_MEL((1293,128,1))
model_chr = CNN_CHR((1293,12,1))
model_mfcc = CRNN_MFCC((1293, 20, 1))

import librosa.display
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
import cv2


for genre in tqdm(["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]):
    for i in range(100):
        output_path=os.path.join(genre,f"{genre}.{i:05d}.txt")
        merge_array = []
        for m in ["mel2_format","chr_format","mfcc_format"]:
            path=os.path.join(m,genre,f"{genre}.{i:05d}.jpg")
            img = cv2.imread(path)
            if (m == "mel2_format"):
                ans, pre = model_mel.predict(img, 30)
            elif (m == "chr_format"):
                ans, pre = model_chr.predict(img, 30)
            else:
                ans, pre = model_mfcc.predict(img, 30)
            pre=pre.flatten()
            merge_array.append(pre)
            output_path = os.path.join(m, output_path)
        merge_array=np.array(merge_array)
        # merge_array=merge_array.reshape((1,30))
        np.savetxt(output_path, merge_array)
        
