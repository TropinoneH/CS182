import librosa 
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

audio_dir = "dataset/audio_format"
beat_dir = "dataset/beat_histogram"
for genre in tqdm(os.listdir(audio_dir), desc="genres"):
    all_files = sorted([files for files in os.listdir(os.path.join(audio_dir, genre)) if
                        not os.path.isdir(files) and files.endswith((".mp3", ".wav", ".WAV", ".MP3"))])
    os.makedirs(os.path.join(beat_dir, genre), exist_ok=True)
    min_fe=0
    max_fe=0
    for file in tqdm(all_files, desc=f"the file in genre-{genre}", leave=False):
        if os.path.exists(os.path.join(beat_dir, genre, file.replace(".wav", ".jpg"))):
            continue
        try:
            y, sr = librosa.load(os.path.join(audio_dir, genre, file))
            tempo,beat_frames=librosa.beat.beat_track(y=y,sr=sr)
            freq=np.histogram(beat_frames.flatten(),bins=5,range=(0,1500))
            np.save(os.path.join(beat_dir, genre, file.replace(".wav", ".npy")),freq[0])
        except Exception as e:
            print(e)
            print(os.path.join(audio_dir, genre, file))