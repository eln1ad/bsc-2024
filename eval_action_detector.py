import json
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from natsort import os_sorted
from pathlib import Path
from normalize import l2_norm

# 0: Define constants
saved_models_dir = Path.cwd().joinpath("saved_models")
data_dir = Path.cwd().joinpath("data")
features_dir = Path("/home/elniad/datasets/boxing/features/rgb")

# 1: Load in the trained action detector

model_name = "action_detector_rgb_window_size_8_window_stride_1_epochs_30"
model_path = saved_models_dir.joinpath(model_name)

model = keras.models.load_model(model_path)
print("Loaded model!")

# 2. Get a random test sample

test_csv = data_dir.joinpath("test_binary_segments_size_8_stride_1_tiou_high_0.5_tiou_low_0.15.csv")
test_df = pd.read_csv(test_csv, index_col=None)
test_sample = test_df.sample(n=1).reset_index(drop=True).to_numpy()[0]

video_name = test_sample[0]
seg_start = test_sample[1]
seg_end = test_sample[2]
gt_start = test_sample[3]
gt_end = test_sample[4]
label = test_sample[5]

# 3. Load features for the segment

video_features_dir = features_dir.joinpath(video_name)
feature_paths = os_sorted(video_features_dir.rglob("*.npy"))

features = []

for feature_path in feature_paths:
    parts = str(feature_path.stem).split("_")
    start, end = int(parts[1]), int(parts[2])
    
    if seg_start <= start and end <= seg_end:
        feature = np.load(feature_path)
        features.append(feature)
        
# take the average
features = np.average(features, axis=0)

# L2 norm
features = l2_norm(features)

# 4. predict with model

features = np.expand_dims(feature, axis=0)

preds = model(features)
pred_labels, pred_centers, pred_lengths = preds
pred_label, pred_center, pred_length = pred_labels[0], pred_centers[0], pred_lengths[0]

# 5. project back the values

if pred_label >= 0.5:
    proj_pred_label = "action"
else:
    proj_pred_label = "background"

seg_length = seg_end - seg_start + 1
seg_center = (seg_end + seg_start) / 2.0

proj_pred_center = seg_center + pred_center
proj_pred_length = seg_length + pred_length

print(proj_pred_center)
print(proj_pred_length)

proj_pred_start = proj_pred_center - proj_pred_length / 2.0
proj_pred_end = proj_pred_center + proj_pred_length / 2.0

print(f">>> EVALUATION OF {str(video_name).upper()} <<<")
print(f"\tTRUE VALUES")
print(f"\t\tlabel: {label}, start: {gt_start}, end: {gt_end}")
print(f"\tPREDICTED VALUES")
print(f"\t\tlabel: {proj_pred_label}, start: {proj_pred_start}, end: {proj_pred_end}")