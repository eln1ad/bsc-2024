import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from natsort import os_sorted # Ez nagyon fontos, mert a fileok különben így lesznek sortolva: [1, 10, 11, 12, 2, 20, 21, 22]



# def preprocess_frame(frame, frame_size):
#     if (frame_size is not None and 
#         isinstance(frame_size, tuple) and 
#         isinstance(frame_size[0], int) and
#         isinstance(frame_size[1], int)):
#         frame = cv2.resize(frame, frame_size)
        
#     if frame.shape[0] == 3:
#         frame = frame.astype(np.float32)
#         frame /= 255
    
#     return frame



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
        label_list = sorted(list(df["label"].unique())) # a sorted kell rá
          
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
                frame = np.load(array_file).astype(np.float32)
                
                # if the frame is RGB then normalization is needed
                # if the frame is FLOW then the values are correct
                if frame.shape[-1] == 3:
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
        "/media/elniad/4tb_hdd/boxing-frames/flow",
        num_frames=8, shuffle=True
    )
    
    possible_labels = sorted(["background", "left_straight", "left_hook", "left_uppercut",
                      "right_straight", "right_hook", "right_uppercut"])

    for frames, label in generator():
        print(frames.shape, label.shape)
        label = possible_labels[np.argmax(label)]
        
        if frames.shape[-1] == 3:
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()

            # Plot each image in its corresponding axis
            for i, ax in enumerate(axes):
                ax.imshow(frames[i])  # You would replace this with your actual image data
                ax.axis('off')  # Turn off axis for cleaner visualization
            
        else:
            fig, axes = plt.subplots(2, 8, figsize=(24, 6))
            
            for i in range(8):
                for j in range(2):
                    axes[j, i].imshow(frames[i, :, :, j])
                    axes[j, i].axis("off")
                    
        plt.title(label, loc="center")
        plt.tight_layout()
        plt.show()
            
        break