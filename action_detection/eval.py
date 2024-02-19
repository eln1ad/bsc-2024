import keras
import pandas as pd
import numpy as np
from natsort import os_sorted
from pathlib import Path
from feature_norm import l2_norm
from utils import load_json, check_file_exists, check_dir_exists
from config_check import get_modality, get_epochs


# 0: load paths, constants


saved_models_dir = Path.cwd().joinpath("saved_models")
data_dir = Path.cwd().joinpath("data")
configs_dir = Path.cwd().joinpath("configs")
detection_data_dir = data_dir.joinpath("detection")


check_dir_exists(saved_models_dir)
check_dir_exists(data_dir)
check_dir_exists(configs_dir)
check_dir_exists(detection_data_dir)


detector_config_json = configs_dir.joinpath("detector.json")
general_config_json = configs_dir.joinpath("general.json")


check_file_exists(detector_config_json)
check_file_exists(general_config_json)


detector_config = load_json(detector_config_json)
general_config = load_json(general_config_json)


modality = get_modality(detector_config)
features_dir = Path(general_config["features_dir"]).joinpath(modality)
epochs = get_epochs(detector_config)


# 1: Load in the trained action detector


model_name = f"detector_{modality}_{epochs}_epochs"
model_path = saved_models_dir.joinpath(model_name)


model = keras.models.load_model(model_path)
print(f"[INFO] Loaded {model_name} model!")


# 2. Get 16 test samples

test_csv = detection_data_dir.joinpath("test.csv")
check_file_exists(test_csv)

test_df = pd.read_csv(test_csv, index_col=None)
samples = test_df.sample(n=16).reset_index(drop=True)


for index, sample in samples.iterrows():
    video_name = sample[0]
    seg_start = sample[1]
    seg_end = sample[2]
    gt_start = sample[3]
    gt_end = sample[4]
    gt_label = sample[5]
    
    feature_paths = os_sorted(features_dir.joinpath(video_name).rglob("*.npy"))

    # 3. Load features for the segment

    features = []

    for feature_path in feature_paths:
        parts = str(feature_path.stem).split("_")
        start, end = int(parts[1]), int(parts[2])
        
        if seg_start <= start and end <= seg_end:
            feature = np.load(feature_path)
            features.append(feature)
            
    # take the average
    features = np.average(features, axis=0)

    # ÖTLET:
    # 1. SEMMI normalizáció
    # 2. mean_std normalizáció
    # features = mean_std_norm(features)
    # 3. l2 normalizáció
    # features = l2_norm(features)

    # 4. Convert as batch and predict

    features = np.expand_dims(feature, axis=0)

    preds = model(features)
    pred_labels, pred_centers, pred_lengths = preds
    pred_label, pred_center, pred_length = float(pred_labels[0].numpy()), float(pred_centers[0].numpy()), float(pred_lengths[0].numpy())

    
    # 5. Convert predictions
    
    if gt_label == 1.0:
        gt_label = "action"
    else:
        gt_label = "background"


    if pred_label >= 0.5:
        label = "action"
    else:
        label = "background"

    seg_length = seg_end - seg_start
    seg_center = (seg_end + seg_start) / 2.0

    pred_center = pred_center * seg_length + seg_center
    pred_length = np.exp(pred_length) * seg_length
    
    pred_start = np.round(pred_center - pred_length / 2.0)
    pred_end = np.round(pred_center + pred_length / 2.0)


    # 6. Print out infos


    print(f">>> EVALUATION OF {str(video_name).upper()} <<<")
    print(f"[TRUE] label: {gt_label}, start: {gt_start}, end: {gt_end}")
    print(f"[PREDICTED] label: {label}, start: {pred_start}, end: {pred_end}\n")