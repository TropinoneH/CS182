import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from NeuralNetwork.CRNN import CRNN_MEL, CRNN_MFCC
from NeuralNetwork.CNN import CNN_CHR

model_mel = CRNN_MEL((1293,128,1))
model_chr = CNN_CHR((1293,12,1))
model_mfcc = CRNN_MFCC((1293, 20, 1))

import numpy as np
from tqdm import tqdm
import cv2

model_mel.load_model("./model/model_mel.h5")
model_chr.load_model("./model/model_chr.h5")
model_mfcc.load_model("./model/model_mfcc.h5")

for genre in tqdm(["blues", "classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]):
    os.makedirs(os.path.join(os.getcwd(), "dataset/merge_format", genre), exist_ok=True)
    for i in range(100):
        output_path=os.path.join(os.getcwd(), "dataset/merge_format", genre,f"{genre}.{i:05d}.txt")
        merge_array = []
        for m in ["mel2_format","chr_format","mfcc_format"]:
            try:
                path=os.path.join(os.getcwd(), "dataset", m,genre,f"{genre}.{i:05d}.jpg")
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)                    
                if img.shape[1] < 1293:
                    gap = 1293 - img.shape[1]
                    img = np.hstack((img,np.zeros((img.shape[0], gap))))
                if (m == "mel2_format"):
                    ans, pre = model_mel.predict(img, 30)
                elif (m == "chr_format"):
                    ans, pre = model_chr.predict(img, 30)
                else:
                    ans, pre = model_mfcc.predict(img, 30)
                pre = pre.flatten()
                merge_array.extend(pre)
            except Exception as e:
                print(e, path)
        if len(merge_array) == 0: continue
        merge_array=np.array(merge_array)
        merge_array=merge_array.reshape((1,30))
        np.savetxt(output_path, merge_array)
        
