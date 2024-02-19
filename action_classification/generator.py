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
    one_hot_encode,
    load_json,
)


def load_label_list(txt_file):
    label_list = []
    with open(txt_file, "r") as file:
        for line in file.readlines():
            label = line.strip()
            label_list.append(label)
    return sorted(label_list)


def get_classification_generator(csv_file, labels_txt_file, features_dir, shuffle=True, norm_method=None):
    check_file_exists(csv_file)
    check_file_exists(labels_txt_file)
    check_dir_exists(features_dir)
    check_norm_method(norm_method)
    
    # the directory name ends with rgb or flow
    modality = str(Path(features_dir).name).lower()
    
    check_modality(modality)
    
    label_list = load_label_list(labels_txt_file)
        
    def generator_func():  
        df = pd.read_csv(csv_file, index_col=None)
          
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        for row in df.itertuples(index=False):
            video_name = row[0]
            seg_start, seg_end = int(row[1]), int(row[2])
            label = row[3]
            
            video_features_dir = Path(features_dir).joinpath(video_name)
            
            # retrive all features between seg_start and seg_end
            file_paths = os_sorted(video_features_dir.rglob("*.npy"))
            
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
            
            yield features, one_hot_encode(label_list, label)
            
    return generator_func

    
        
if __name__ == "__main__":
    configs_dir = Path.cwd().joinpath("configs")
    action_classification_data_dir = Path.cwd().joinpath("data", "action_classification")
    
    modality = "rgb"
    
    paths_config = load_json(configs_dir.joinpath("paths.json"))
    
    features_dir = Path(paths_config["features_dir"]).joinpath(modality)
    
    generator = get_classification_generator(
        action_classification_data_dir.joinpath("train.csv"),
        action_classification_data_dir.joinpath("label_list.txt"),
        features_dir,
        shuffle=True
    )
    
    num_labels = len(load_label_list(action_classification_data_dir.joinpath("label_list.txt")))
    
    output_signature = (
        tf.TensorSpec(shape=(1024,), dtype=tf.float32),
        tf.TensorSpec(shape=(num_labels,), dtype=tf.float32),
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_signature=output_signature).batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    for i, (x, y) in enumerate(dataset):
        print(x.shape, y.shape)
        print(x)
        print(y)
        break