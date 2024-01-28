import os
import shutil
os.makedirs('./dataset/beat_train',exist_ok=True)
os.makedirs('./dataset/beat_test',exist_ok=True)
dir='./dataset/beat_histogram'
for genre in (os.listdir(dir)):
    print(genre)
    count=0
    for file in (os.listdir(os.path.join(dir,genre))):
        if count<75:
            if not os.path.exists(os.path.join('./dataset/beat_train',genre)):
                os.makedirs(os.path.join('./dataset/beat_train',genre))
            root=os.path.join(dir,genre,file)
            new_root=os.path.join('./dataset/beat_train',genre,file)
            shutil.copy(root,new_root)
        else:
            if not os.path.exists(os.path.join('./dataset/beat_test',genre)):
                os.makedirs(os.path.join('./dataset/beat_test',genre))
            root=os.path.join(dir,genre,file)
            new_root=os.path.join('./dataset/beat_test',genre,file)
            shutil.copy(root,new_root)
        count+=1