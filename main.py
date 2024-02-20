import cv2
import numpy as np
import keras
from pathlib import Path
from utils import load_json
from extract_features import get_feature_extractor
from action_classification.generator import load_label_list


# 1. Get a video
# 2. Break to video into sliding windows
# 3. Extract features with feature extractor (length=8, stride=1)
# 4. Input the features into the detector model to get predictions (action/background, delta_center, delta_length)
# 5. Continue with the action proposals only, refine their boundaries with delta_center, delta_length
# 6. Get the features of the refined window and input them into the classifier to predict the class
# 7. Filter proposals with NMS


def get_rgb_frames(path: str, frame_size):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame /= 255.0
        frames.append(frame)
    return np.array(frames)


def get_window_intervals(num_frames, sizes = [8, 10, 12, 14, 16], overlap = 0.8):
    windows = []
    for size in sizes:
        stride = round((1 - overlap) * size)
        starts = np.arange(0, num_frames - size + 1, stride)
        ends = starts + size
        windows.extend(list(zip(starts, ends)))
    windows = np.clip(windows, a_min=0, a_max=num_frames-1)
    return windows



# def get_window_indices(sample_count, start, end):
#     n = end - start
#     sample_rate = int(n / sample_count)
#     indices = np.arange(start, end)
    
#     out_indices = []
    
#     if sample_rate >= 2:
#         for i in range(sample_count):
#             group = np.arange(i * sample_rate, (i + 1) * sample_rate)
#             i = np.random.choice(group, 1, replace=False).item()
#             out_indices.append(indices[i])
#         if (n - sample_count * sample_rate) > 0:
#             left_overs = np.arange(sample_count * sample_rate, n)
#             i = np.random.choice(left_overs, 1, replace=False).item()
#             out_indices.append(indices[i])
#             out_indices = np.sort(np.random.choice(out_indices, sample_count, replace=False))
#     else:
#         if n < sample_count:
#             out_indices = np.sort(np.random.choice(indices, sample_count, replace=True))
#         else:
#             out_indices = np.sort(np.random.choice(indices, sample_count, replace=False))
            
#     return out_indices
    
    


if __name__ == "__main__":
    video_name = "video_000351.mp4"
    frame_size = (112, 112)
    c3d_capacity = 8
    modality = "rgb"
    
    configs_dir = Path.cwd().joinpath("configs")
    saved_models_dir = Path.cwd().joinpath("saved_models")
    paths_config = load_json(configs_dir.joinpath("paths.json"))
    video_path = Path(paths_config["videos_dir"]).joinpath(video_name)
    
    frames = get_rgb_frames(str(video_path), frame_size)
    
    video_classifier_name = f"C3D_binary_{modality}_10_epochs"
    feature_extractor = get_feature_extractor(video_classifier_name)
    
    action_detector_name = f"action_detector_{modality}_50_epochs"
    action_detector = keras.models.load_model(saved_models_dir.joinpath(action_detector_name))
    
    action_classifier_name = f"action_classifier_{modality}_50_epochs"
    action_classifier = keras.models.load_model(saved_models_dir.joinpath(action_classifier_name))
    
    window_size = 8
    window_stride = 1
    
    features_dict = {}
    
    label_list_txt = Path.cwd().joinpath("data", "action_classification", "label_list.txt")
    label_list = load_label_list(label_list_txt)
    
    for i in range(0, frames.shape[0] - window_size + 1, window_stride):
        start = i * window_stride
        end = start + window_size
        indices = np.arange(start, end)
        input_batch = np.expand_dims(frames[indices], axis=0)
        features = feature_extractor(input_batch)[0]
        
        features_dict[f"{start}_{end}"] = features
        
    # Create sliding windows
    
    window_intervals = get_window_intervals(frames.shape[0], sizes=[8, 10, 12, 14, 16], overlap=0.8)
    
    for window_interval in window_intervals:
        # Load in corresponding features
        window_features = []
        for k in features_dict.keys():
            parts = k.split("_")
            start, end = int(parts[0]), int(parts[1])
            if window_interval[0] <= start and end <= window_interval[1]:
                window_features.append(features_dict[k])
        window_features = np.average(window_features, axis=0)
        window_features = np.expand_dims(window_features, axis=0)
        # Use the features as input to the action_detector
        pred = action_detector(window_features)
        pred_label, pred_center, pred_length = float(pred[0][0].numpy()), float(pred[1][0].numpy()), float(pred[2][0].numpy())
        # Do nothing if background
        if pred_label <= 0.5:
            continue
        # Use the regressed coordinates to calculate new window
        # Load in the features with the new coordinates and predict class
        else:
            window_length = window_interval[1] - window_interval[0]
            window_center = (window_interval[1] + window_interval[0]) / 2.0
            
            pred_center = pred_center * window_length + window_center
            pred_length = np.exp(pred_length) * window_length
            
            pred_start = np.round(pred_center - pred_length / 2.0)
            pred_start = np.clip(pred_start, a_min=0, a_max=window_interval[0])
            
            pred_end = np.round(pred_center + pred_length / 2.0)
            pred_end = np.clip(pred_end, a_min=0, a_max=window_interval[1])
            
            regressed_window_features = []
            
            for k in features_dict.keys():
                parts = k.split("_")
                start, end = int(parts[0]), int(parts[1])
                if pred_start <= start and end <= pred_end:
                    regressed_window_features.append(features_dict[k])
            
            regressed_window_features = np.average(regressed_window_features, axis=0)
            regressed_window_features = np.expand_dims(regressed_window_features, axis=0)
            
            pred = action_classifier(regressed_window_features)[0]
            pred_confidence = round(np.max(pred), 4)
            pred_label = label_list[np.argmax(pred)]
            
            print(f">>> PREDICTION FOR {Path(video_name).stem} <<<")
            print(f"label: {pred_label}, confidence: {pred_confidence}, start: {pred_start}, end: {pred_end}\n")