import numpy as np
import pandas as pd
from pathlib import Path
from natsort import os_sorted # Ez nagyon fontos, mert a fileok különben így lesznek sortolva: [1, 10, 11, 12, 2, 20, 21, 22]
from feature_norm import l2_norm, mean_std_norm
import tensorflow as tf


# a tensorflow-s generatornak nem lehet parametere, ezért egy nested függvényt írtam
def get_detection_generator(csv_file, video_features_dir, shuffle=True):
    if not Path(csv_file).exists():
            raise ValueError("[ERROR] 'csv_file' does not exist!")
        
    if not Path(video_features_dir).exists():
        raise ValueError("[ERROR] 'video_features_dir' does not exist!")
    
    # the directory name ends with rgb or flow
    modality = str(Path(video_features_dir).name).lower()
    
    if modality not in ["rgb", "flow"]:
        raise ValueError(
            "[ERROR] 'modality' can only be rgb or flow!\n"
            "HINT: You might need to rename your directory to rgb or flow!\n"
            "/home/elniad/datasets/boxing/frames/rgb"
        )
        
    def generator_func():  
        df = pd.read_csv(csv_file, index_col=None)
          
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        for row in df.itertuples(index=False):
            video_name = row[0]
            seg_start, seg_end = int(row[1]), int(row[2])
            label = row[5]
            delta_center = row[7]
            delta_length = row[8]
            
            video_dir = Path(video_features_dir).joinpath(video_name)
            
            # retrive all features between seg_start and seg_end
            file_paths = os_sorted(video_dir.rglob("*.npy"))
            
            features = []
            
            for file_path in file_paths:
                parts = str(file_path.stem).split("_")
                start, end = int(parts[1]), int(parts[2])
                
                if seg_start <= start and end <= seg_end:
                    feature = np.load(file_path)
                    features.append(feature)
                    
            features = np.average(features, axis=0)
            
            # ÖTLET:
            # 1. SEMMI normalizáció
            # 2. mean_std normalizáció
            # features = mean_std_norm(features)
            # 3. l2 normalizáció
            # features = l2_norm(features)
            
            # Most semmilyen normalizációt nem használok
            # features = l2_norm(features)
            
            yield features, (np.array([label]), np.array([delta_center]), np.array([delta_length]))
            
    return generator_func
    
        
if __name__ == "__main__":
    data_dir = Path.cwd().joinpath("data")
    detection_data_dir = data_dir.joinpath("detection")
    
    modality = "rgb"
    features_dir = f"/home/elniad/datasets/boxing/features/{modality}"
    
    generator = get_detection_generator(
        detection_data_dir.joinpath("train.csv"), 
        features_dir,
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

    for i, (x, y) in enumerate(dataset):
        print(x.shape)
        label, center, length = y
        # print(y_label.shape)
        # print(y_center.shape)
        # print(y_length.shape)
        # print(y_label)
        # print(y_center)
        # print(y_length)
        print(label, center, length)
        print(np.max(x[0]), np.min(x[0]))
        break