import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from natsort import os_sorted # Ez nagyon fontos, mert a fileok különben így lesznek sortolva: [1, 10, 11, 12, 2, 20, 21, 22]


# a tensorflow-s generatornak nem lehet parametere, ezért egy nested függvényt írtam
def get_frames_label_generator(csv_file, video_frames_dir, num_frames=None, shuffle=True):
    if not Path(csv_file).exists():
            raise ValueError("'csv_file' does not exist!")
        
    if not Path(video_frames_dir).exists():
        raise ValueError("'video_frames_dir' does not exist!")
    
    if num_frames is None:
        raise ValueError("'num_frames' must have a value!")
    
    if not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("'num_frames' must be a positive integer!")
        
    def frames_label_generator():   
        df = pd.read_csv(csv_file, index_col=None)
        label_list = list(df["label"].unique())
          
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        for row in df.itertuples(index=False):
            video_name = row[0]
            seg_start, seg_end = int(row[1]), int(row[2])
            label = row[-2]
            
            video_dir = Path(video_frames_dir).joinpath(video_name)
            array_files = os_sorted(video_dir.rglob("*.npy"))
            array_files = array_files[seg_start : seg_end]
            
            frames = []
            
            for array_file in array_files:
                frame = np.load(array_file).astype(float)
                frame /= 255
                frames.append(frame)
                
            frames = np.array(frames)
            
            # trimming last frames
            if frames.shape[0] > num_frames:
                frames = frames[:num_frames]
                
            # padding with last frame
            if frames.shape[0] < num_frames:
                pad_amnt = num_frames - frames.shape[0] # frames.shape => (8, 112, 112, 3)
                last_frame = np.copy(frames[-1])
                last_frame = np.expand_dims(last_frame, axis=0) # az append miatt kell
                
                for i in range(pad_amnt):
                    frames = np.append(frames, last_frame, axis=0)
                
            one_hot = np.zeros(len(label_list))
            one_hot[label_list.index(label)] = 1.0
                    
            yield frames, one_hot
            
    return frames_label_generator
    
        
if __name__ == "__main__":
    generator = get_frames_label_generator(
        "train_segments_size_8_stride_1.csv", 
        "/media/elniad/4tb_hdd/boxing-frames/rgb",
        num_frames=8, shuffle=True
    )
    
    i = 1

    for frames, label in generator():
        print(frames.shape, label.shape)
        if i == 5:
            break
        else:
            i += 1