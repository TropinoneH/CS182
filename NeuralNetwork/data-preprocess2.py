import os
import random

import librosa.display
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
import cv2

for genre in tqdm(["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]):
    for i in range(100):
        output_path=os.path.join(m,genre,f"{genre}.{i%05d}.txt")
        merge_array = []
        for m in ["mel2_format","chr_format","mfcc_format"]
            path=os.path.join(m,genre,f"{genre}.{i%05d}.jpg")
            img = cv2.imread(path)
            if (m == "mel2_format"):
                ans, pre = model_mel.predict(mel_pic, 30)
            elif (m == "chr_format"):
                ans, pre = model_chr.predict(harmonic_features, 30)
            else:
                ans, pre = model_mfcc.predict(mfccs, 30)
            pre=pre.flatten()
            merge_array.extend(pre)
        merge_array=np.array(merge_array)
        merge_array=merge_array.reshape((1,30))
        np.savetxt(output_path, merge_array)
        
