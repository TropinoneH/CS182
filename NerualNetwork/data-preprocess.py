import os
import random

import librosa.display
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm

audio_dir = "/home/mspt5/Documents/homework/IML/project/dataset/audio_format"
mel_dir = "/home/mspt5/Documents/homework/IML/project/dataset/mel_format"
for genre in tqdm(os.listdir(audio_dir), desc="genre"):
    all_files = sorted([files for files in os.listdir(os.path.join(audio_dir, genre)) if
                        not os.path.isdir(files) and files.endswith((".mp3", ".wav", ".WAV", ".MP3"))])
    os.makedirs(os.path.join(mel_dir, genre), exist_ok=True)
    for file in tqdm(all_files, desc=f"the file in genre-{genre}", leave=False):
        if os.path.exists(os.path.join(mel_dir, genre, file.replace(".wav", ".jpg"))):
            continue
        try:
            y, sr = librosa.load(os.path.join(audio_dir, genre, file))
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            # librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.savefig(os.path.join(mel_dir, genre, file.replace(".wav", ".jpg")), bbox_inches=None, pad_inches=0)
            # plt.close()
            for i in range(5):
                pic = librosa.power_to_db(spectrogram, ref=np.max)
                start_pos = random.randint(0, pic.shape[1] - 128 - 1)
                mpimg.imsave(os.path.join(mel_dir, genre, file.replace(".wav", f".{i + 1}.jpg")),
                             pic[:, start_pos:start_pos + 128], cmap="gray")
        except Exception as e:
            print(e)
            print(os.path.join(audio_dir, genre, file))
