import cv2
import numpy as np
import keras
from pathlib import Path
from utils import load_json
from extract_features import get_feature_extractor
from action_classification.generator import load_label_list


# 1. Get a video
# 2. Get sliding windows for the video
# 3. Extract features for the whole video, with a fixed range (length=8, stride=1)
# 4. Input the features corresponding to a sliding window to the action detector model to get stage 1 predictions (action/background, delta_center, delta_length)
# 5. Continue with those sliding windows that are actions
# 6. Calculate the new start and end coordinates of those windows, using delta_center and delta_length
# 7. Using this new regressed window load in the necessary features
# 8. Input features to the action classifier model to predict the class of the action
# 9. Filter the output windows with the NMS algorithm
# 10. Display the information of the retained windows (start, end, label, confidence)


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


def NMS(segments, confidences, tiou_threshold=0.2):
    starts = segments[:, 0]
    ends = segments[:, 1]
    
    # sort the scores by indices descending (highest first)
    indices = np.argsort(confidences)[::-1]
    areas = (ends - starts).astype(float)

    filtered = []
    
    while len(indices) > 0:
        best_idx = indices[0]
        filtered.append(best_idx)
        
        indices = indices[1:] # delete first (best) item

        # get intersection between the best box and the rest
        intersection_starts = np.maximum(starts[best_idx], starts[indices])
        intersection_ends = np.minimum(ends[best_idx], ends[indices])

        # calculate tiou
        intersections = np.maximum(0., intersection_ends - intersection_starts)
        unions = areas[best_idx] + areas[indices] - intersections
        tious = intersections / unions

        # remove those segments that have high tiou with best
        indices = indices[np.nonzero(tious <= tiou_threshold)[0]]

    # return indices of filtered segments
    return filtered
    

if __name__ == "__main__":
    configs_dir = Path.cwd().joinpath("configs")
    c3d_config = load_json(configs_dir.joinpath("C3D.json"))
    paths_config = load_json(configs_dir.joinpath("paths.json"))
    action_detector_config = load_json(configs_dir.joinpath("action_detector.json"))
    action_classifier_config = load_json(configs_dir.joinpath("action_classifier.json"))
    
    
    video_name = "video_000351.mp4"
    frame_size = (c3d_config["image_width"], c3d_config["image_height"])
    capacity = c3d_config["capacity"]
    modality = c3d_config["modality"]
    nms_threshold = 0.2
    
    
    saved_models_dir = Path.cwd().joinpath("saved_models")
    video_path = Path(paths_config["videos_dir"]).joinpath(video_name)
    
    frames = get_rgb_frames(str(video_path), frame_size)
    
    video_classifier_name = f"C3D_binary_{modality}_10_epochs"
    feature_extractor = get_feature_extractor(video_classifier_name)
    
    action_detector_name = f"action_detector_{modality}_50_epochs"
    action_detector = keras.models.load_model(saved_models_dir.joinpath(action_detector_name))
    
    action_classifier_name = f"action_classifier_{modality}_50_epochs"
    action_classifier = keras.models.load_model(saved_models_dir.joinpath(action_classifier_name))
    
    
    # window_size used for extracting features
    window_size = c3d_config["capacity"]
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
        
    window_sizes = [8, 10, 12, 14, 16]
    window_intervals = get_window_intervals(frames.shape[0], sizes=window_sizes, overlap=0.8)
    
    predictions = []
    
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
            
            window_features = []
            
            for k in features_dict.keys():
                parts = k.split("_")
                start, end = int(parts[0]), int(parts[1])
                if pred_start <= start and end <= pred_end:
                    window_features.append(features_dict[k])
            
            window_features = np.average(window_features, axis=0)
            window_features = np.expand_dims(window_features, axis=0)
            
            pred = action_classifier(window_features)[0]
            pred_confidence = round(np.max(pred), 4)
            pred_label = label_list[np.argmax(pred)]
            
            predictions.append([pred_start, pred_end, pred_confidence, pred_label])
            
    predictions = np.array(predictions)
    # casting is needed, because there is a string column inside predictions
    indices = NMS(predictions[:, :2].astype(float), predictions[:, 2].astype(float), tiou_threshold=nms_threshold)
    predictions = predictions[indices]
    
    for prediction in predictions:
        pred_label = prediction[3]
        pred_start, pred_end = prediction[0], prediction[1]
        pred_confidence = prediction[2]
        print(f">>> PREDICTION FOR {Path(video_name).stem} <<<")
        print(f"label: {pred_label}, confidence: {pred_confidence}, start: {pred_start}, end: {pred_end}\n")