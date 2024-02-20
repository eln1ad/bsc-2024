import keras
import pandas as pd
import numpy as np
from natsort import os_sorted
from pathlib import Path
from feature_norm import l2_norm, mean_std_norm, min_max_norm
from utils import load_json, check_file_exists, check_dir_exists
from config_check import get_modality, get_epochs
from action_classification.generator import load_label_list


# 0: load paths, constants
saved_models_dir = Path.cwd().joinpath("saved_models")
check_dir_exists(saved_models_dir)


configs_dir = Path.cwd().joinpath("configs")
check_dir_exists(configs_dir)


data_dir = Path.cwd().joinpath("data")
check_dir_exists(data_dir)


action_classification_data_dir = data_dir.joinpath("action_classification")
check_dir_exists(action_classification_data_dir)


classifier_json = configs_dir.joinpath("action_classifier.json")
check_file_exists(classifier_json)


paths_json = configs_dir.joinpath("paths.json")
check_file_exists(paths_json)


model_config = load_json(classifier_json)
paths_config = load_json(paths_json)


modality = get_modality(model_config)
epochs = get_epochs(model_config)


features_dir = Path(paths_config["features_dir"]).joinpath(modality)
check_dir_exists(features_dir)


# 1: Load trained detector
model_name = f"action_classifier_{modality}_{epochs}_epochs"
saved_model_path = saved_models_dir.joinpath(model_name)
check_dir_exists(saved_model_path)


model = keras.models.load_model(saved_model_path)
print(f"[INFO] Loaded {model_name} model!")


test_csv = action_classification_data_dir.joinpath("test.csv")
check_file_exists(test_csv)


label_list_txt = action_classification_data_dir.joinpath("label_list.txt")
check_file_exists(label_list_txt)


label_list = load_label_list(label_list_txt)


def evaluate(num_samples = 16, norm_method = None):
    if num_samples <= 0:
        raise ValueError("[ERROR] 'num_samples' must be greater than 0!")
    
    if norm_method not in ["l2", "mean_std", "min_max", None]:
        raise ValueError("[ERROR] 'norm_method' can only be 'l2', 'mean_std', 'min_max' or None!")
    
    samples = pd.read_csv(test_csv, index_col = None).sample(n = num_samples)

    for _, sample in samples.iterrows():
        video_name = sample[0]
        seg_start = sample[1]
        seg_end = sample[2]
        gt_label = sample[3]
        
        video_feaures_dir = features_dir.joinpath(video_name)
        check_dir_exists(video_feaures_dir)
        
        feature_paths = os_sorted(features_dir.joinpath(video_name).rglob("*.npy"))

        # 3. Load features
        features = []

        for feature_path in feature_paths:
            parts = str(feature_path.stem).split("_")
            start, end = int(parts[1]), int(parts[2])
            
            if seg_start <= start and end <= seg_end:
                feature = np.load(feature_path)
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
        
        features = np.expand_dims(feature, axis=0) # Convert to batch

        # 4. Predict
        pred = model(features)[0]
        
        # 5. Convert predictions
        idx = np.argmax(pred)
        confidence = round(np.max(pred), 4)
        pred_label = label_list[idx]
        

        # 6. Print out infos
        print(f">>> EVALUATION OF {str(video_name).upper()} <<<")
        print(f"[TRUE] label: {gt_label}")
        print(f"[PREDICTED] label: {pred_label}, confidence: {confidence}\n")
        
        
if __name__ == "__main__":
    evaluate(num_samples=16, norm_method=None)