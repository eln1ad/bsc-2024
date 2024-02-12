import numpy as np
import pandas as pd
from pathlib import Path
from natsort import os_sorted # Ez nagyon fontos, mert a fileok különben így lesznek sortolva: [1, 10, 11, 12, 2, 20, 21, 22]
from normalize import l2_norm
import tensorflow as tf


# a tensorflow-s generatornak nem lehet parametere, ezért egy nested függvényt írtam
def get_features_label_generator(csv_file, video_features_dir, shuffle=True):
    if not Path(csv_file).exists():
            raise ValueError("'csv_file' does not exist!")
        
    if not Path(video_features_dir).exists():
        raise ValueError("'video_features_dir' does not exist!")
    
    # the directory name ends with rgb or flow
    modality = str(Path(video_features_dir).name).lower()
    
    if modality not in ["rgb", "flow"]:
        raise ValueError(
            "'modality' can only be rgb or flow!\n"
            "HINT: You might need to rename your directory to rgb or flow!\n"
            "/home/elniad/datasets/boxing/frames/rgb"
        )
        
    def features_label_generator():   
        df = pd.read_csv(csv_file, index_col=None)
          
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        for row in df.itertuples(index=False):
            # start_time = time.time()
            
            video_name = row[0]
            seg_start, seg_end = int(row[1]), int(row[2])
            gt_start, gt_end = int(row[3]), int(row[4])
            label = row[5]
            
            video_dir = Path(video_features_dir).joinpath(video_name)
            
            # retrive all features between seg_start and seg_end
            fpaths = os_sorted(video_dir.rglob("*.npy"))
            
            X = []
            
            for fpath in fpaths:
                parts = str(fpath.stem).split("_")
                start, end = int(parts[1]), int(parts[2])
                
                if seg_start <= start and end <= seg_end:
                    feature = np.load(fpath)
                    X.append(feature)
                    
            # take the average
            X = np.average(X, axis=0)
            
            # L2 norm
            X = l2_norm(X)
            
            if label == "action":
                y_label = np.array([1.0], dtype=np.float32)
            else:
                y_label = np.array([0.0], dtype=np.float32)
                
            # encode center, length
            gt_length = gt_end - gt_start + 1
            seg_length = seg_end - seg_start + 1
            
            gt_center = (gt_end + gt_start) / 2
            seg_center = (seg_end + seg_start) / 2
            
            y_center = np.array([(gt_center - seg_center) / seg_length], dtype=np.float32)
            y_length = np.array([np.log(gt_length / seg_length)], dtype=np.float32)
            
            yield X, (y_label, y_center, y_length)
            
    return features_label_generator
    
        
if __name__ == "__main__":
    data_dir = Path.cwd().joinpath("data")
    
    
    generator = get_features_label_generator(
        data_dir.joinpath("/home/elniad/bsc-2024/data/train_binary_segments_size_8_stride_1_tiou_high_0.5_tiou_low_0.15.csv"), 
        "/home/elniad/datasets/boxing/features/rgb",
        shuffle=True
    )
    
    output_signature = (
        tf.TensorSpec(shape=(1024,), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )
    )
    
    
    
    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_signature=output_signature).batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    for X, y in dataset:
        print(X.shape)
        y_label, y_center, y_length = y
        print(y_label.shape)
        print(y_center.shape)
        print(y_length.shape)
        print(y_center)
        print(y_length)
        break