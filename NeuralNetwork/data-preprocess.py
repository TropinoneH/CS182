import os
import random

import librosa.display
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm

audio_dir = "dataset/audio_format"
# mel_dir = "dataset/mel_format"
mfcc_dir = "dataset/mfcc_format"
chr_dir = "dataset/chr_format"
for genre in tqdm(os.listdir(audio_dir), desc="genre"):
    all_files = sorted([files for files in os.listdir(os.path.join(audio_dir, genre)) if
                        not os.path.isdir(files) and files.endswith((".mp3", ".wav", ".WAV", ".MP3"))])
    os.makedirs(os.path.join(mfcc_dir, genre), exist_ok=True)
    for file in tqdm(all_files, desc=f"the file in genre-{genre}", leave=False):
        if os.path.exists(os.path.join(mfcc_dir, genre, file.replace(".wav", ".jpg"))):
            continue
        try:
            y, sr = librosa.load(os.path.join(audio_dir, genre, file))
            # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            # harmonic = librosa.effects.harmonic(y) ##
            # harmonic_features = librosa.feature.chroma_cqt(y=harmonic, sr=sr) ##
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            # librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
            # spectrogram.colorbar(format='%+2.0f dB')
            # spectrogram.savefig(os.path.join(mel_dir, genre, file.replace(".wav", ".jpg")), bbox_inches=None, pad_inches=0)
            # spectrogram.close()
            # for i in range(5):
                # pic = librosa.power_to_db(spectrogram, ref=np.max)
                # start_pos = random.randint(0, harmonic_features.shape[1] - 128 - 1)
                # mpimg.imsave(os.path.join(mel_dir, genre, file.replace(".wav", f".{i + 1}.jpg")),
                #              pic[:, start_pos:start_pos + 128], cmap="gray")
            mpimg.imsave(os.path.join(mfcc_dir, genre, file.replace(".wav", f".jpg")),
                             mfccs, cmap="gray")  ##
        except Exception as e:
            print(e)
            print(os.path.join(audio_dir, genre, file))

# test_dir = "../dataset/mel_test"
# for genre in tqdm(os.listdir(audio_dir), desc="genre"):
#     all_files = sorted([files for files in os.listdir(os.path.join(audio_dir, genre)) if
#                         not os.path.isdir(files)])
#     os.makedirs(os.path.join(test_dir, genre), exist_ok=True)
#     for file in tqdm(all_files, desc=f"the file in genre-{genre}", leave=False):
#         if os.path.exists(os.path.join(mel_dir, genre, file.replace(".wav", ".jpg"))):
#             continue
#         try:
#             y, sr = librosa.load(os.path.join(audio_dir, genre, file))
#             spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#             pic = librosa.power_to_db(spectrogram, ref=np.max)
#             start_pos = random.randint(0, pic.shape[1] - 128 - 1)
#             mpimg.imsave(os.path.join(test_dir, genre, file.replace(".wav", f".jpg")),
#                          pic[:, start_pos:start_pos + 128], cmap="gray")
#         except Exception as e:
#             print(e)
#             print(os.path.join(audio_dir, genre, file))
