import numpy as np
import pandas as pd
from pathlib import Path
from natsort import os_sorted # Ez nagyon fontos, mert a fileok különben így lesznek sortolva: [1, 10, 11, 12, 2, 20, 21, 22]
from feature_norm import l2_norm, mean_std_norm, min_max_norm
import tensorflow as tf

from utils import (
    check_dir_exists, 
    check_file_exists,
    check_modality,
    check_norm_method,
    load_json,
)


# a tensorflow-s generatornak nem lehet parametere, ezért egy nested függvényt írtam
def get_action_detection_generator(csv_file, video_features_dir, shuffle=True, norm_method=None):
    check_file_exists(csv_file)
    check_dir_exists(video_features_dir)
    check_norm_method(norm_method)
    
    # the directory name ends with rgb or flow
    modality = str(Path(video_features_dir).name).lower()
    
    check_modality(modality)
        
    def generator_func():  
        df = pd.read_csv(csv_file, index_col=None)
          
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        for row in df.itertuples(index=False):
            video_name = row[0]
            seg_start, seg_end = int(row[1]), int(row[2])
            label = row[5]
            delta_center = row[8]
            delta_length = row[9]
            
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
            
            if norm_method == "l2":
                features = l2_norm(features)
            elif norm_method == "mean_std":
                features = mean_std_norm(features)
            elif norm_method == "min_max":
                features = min_max_norm(features)
            else:
                pass
            
            yield features, (np.array([label]), np.array([delta_center]), np.array([delta_length]))
            
    return generator_func

    
        
if __name__ == "__main__":
    data_dir = Path.cwd().joinpath("data")
    action_detection_data_dir = data_dir.joinpath("action_detection")
    configs_dir = Path.cwd().joinpath("configs")
    
    modality = "rgb"
    
    paths_config = load_json(configs_dir.joinpath("paths.json"))
    
    features_dir = Path(paths_config["features_dir"]).joinpath(modality)
    
    generator = get_action_detection_generator(
        action_detection_data_dir.joinpath("train.csv"), 
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
        print(label, center, length)
        print(np.max(x[0]), np.min(x[0]))
        break