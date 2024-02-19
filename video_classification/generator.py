import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from natsort import os_sorted # Ez nagyon fontos, mert a fileok különben így lesznek sortolva: [1, 10, 11, 12, 2, 20, 21, 22]
from utils import (
    one_hot_encode,
    check_file_exists,
    check_dir_exists,
    check_modality,
)


# a tensorflow-s generatornak nem lehet parametere, ezért egy nested függvényt írtam
def get_video_classification_generator(csv_file, video_frames_dir, num_frames=None, task="binary", shuffle=True):
    check_file_exists(csv_file)
    check_dir_exists(video_frames_dir)
    
    if num_frames is None:
        raise ValueError("[ERROR] 'num_frames' must have a value!")
    
    if not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("[ERROR] 'num_frames' must be a positive integer!")
    
    
    if task not in ["binary", "multi"]:
        raise ValueError("[ERROR] 'task' can only have the following values: (binary, multi)")
    
    # the directory name ends with rgb or flow
    modality = str(Path(video_frames_dir).name).lower()
    
    check_modality(modality)
        
    def generator_func():   
        df = pd.read_csv(csv_file, index_col=None)
        label_list = sorted(list(df["label"].unique())) # a sorted kell rá
          
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        for row in df.itertuples(index=False):
            
            video_name = row[0]
            seg_start, seg_end = int(row[1]), int(row[2])
            label = row[5]
            
            video_dir = Path(video_frames_dir).joinpath(video_name)
            
            if modality == "rgb":
                fpaths = os_sorted(video_dir.rglob("*.png"))
            else:
                fpaths = os_sorted(video_dir.rglob("*.npy"))
                
            fpaths = fpaths[seg_start : seg_end]                
            frames = []
            
            for fpath in fpaths:
                if modality == "rgb":
                    frame = cv2.imread(str(fpath))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32)
                    frame /= 255.0
                else:
                    frame = np.load(fpath)
                
                frames.append(frame)
                
            frames = np.array(frames)
            
            # let's sample randomly
            frame_count = frames.shape[0]
            # pl: frame_count = 14, num_frames = 8
            sample_rate = int(frame_count / num_frames)
            # this will be more complicated
            if sample_rate >= 2:
                indices = []
                for i in range(num_frames):
                    group = np.arange(i * sample_rate, (i + 1) * sample_rate)
                    # get one idx from the current group
                    idx = np.random.choice(group, 1, replace=False).item()
                    indices.append(idx)
                # THIS STILL NEEDS TESTING CAUSE THERE ARE NO ITEMS
                # LARGER THAN 16 FRAMES IN THE CURRENT SETS
                # there might be left over items at the end
                if (frame_count - num_frames * sample_rate) > 0:
                    left_overs = np.arange(num_frames * sample_rate, frame_count)
                    idx = np.random.choice(left_overs, 1, replace=False).item()
                    # now there is an extra item, so we have to shuffle and random sample
                    # again
                    indices.append(idx)
                    indices = np.sort(np.random.choice(indices, num_frames, replace=False))
            else:
                indices = np.arange(frame_count)
                
                # if there is not enough frames then sample
                # with replacement
                if frame_count < num_frames:
                    indices = np.sort(np.random.choice(indices, num_frames, replace=True))
                else:
                    indices = np.sort(np.random.choice(indices, num_frames, replace=False))
                    
            # print(indices)
                
            # sample only the frames specified by indices
            frames = frames[indices]
            
            # if the task is binary classification then the
            # labels are already encoded as 0 and 1, but they
            # have to be converted as arrays, because of tensorflow  
            if task == "binary":
                yield frames, np.array([label])
            # if the task is multiclass classification then the
            # labels will be one-hot encoded on the fly
            else:
                one_hot = one_hot_encode(label_list, label)
                yield frames, one_hot
            
    return generator_func
    
        
if __name__ == "__main__":
    TASK = "binary"
    data_dir = Path.cwd().joinpath("data")
    video_classifcation_data_dir = data_dir.joinpath("video_classification")
    
    
    generator = get_video_classification_generator(
        video_classifcation_data_dir.joinpath(f"{TASK}_train.csv"),
        Path("/home/elniad/datasets/boxing/frames/rgb"),
        num_frames=8, shuffle=True,
        task="binary"
    )
    
    for x, y in generator():
        print(x.shape, y.shape)
        break

    # for frames, label in generator():
    #     print(frames.shape, label.shape)
    #     label = possible_labels[np.argmax(label)]
        
    #     if frames.shape[-1] == 3:
    #         fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    #         axes = axes.flatten()

    #         # Plot each image in its corresponding axis
    #         for i, ax in enumerate(axes):
    #             ax.imshow(frames[i])  # You would replace this with your actual image data
    #             ax.axis('off')  # Turn off axis for cleaner visualization
            
    #     else:
    #         fig, axes = plt.subplots(2, 8, figsize=(24, 6))
            
    #         for i in range(8):
    #             for j in range(2):
    #                 axes[j, i].imshow(frames[i, :, :, j])
    #                 axes[j, i].axis("off")
                    
    #     plt.title(label, loc="center")
    #     plt.tight_layout()
    #     plt.show()
        
    #     break