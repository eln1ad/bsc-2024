import cv2
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from natsort import os_sorted
from utils import check_dir_exists


# Which layer should be used in C3D? relu_fc_6
def get_feature_extractor(model_name, last_layer_name="relu_fc_6"):
    saved_model_path = Path.cwd().joinpath("saved_models").joinpath(model_name)
    
    if not saved_model_path.exists():
        raise ValueError(f"[ERROR] {model_name} does not exist inside saved_models directory!")
    
    model = keras.models.load_model(saved_model_path)
    last_layer = model.get_layer(name=last_layer_name)
    model = tf.keras.Model(inputs=model.inputs, outputs=last_layer.output)
    return model


def run_extraction(dataset_dir, model_name, window_length = 8, window_stride = 1):
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
        
    feature_extractor = get_feature_extractor(model_name)
    
    if "rgb" in model_name:
        modality = "rgb"
        glob_pattern = "*.png"
    elif "flow" in model_name:
        modality = "flow"
        glob_pattern = "*.npy"
    else:
        raise ValueError    
        
    frames_dir = dataset_dir.joinpath("frames", modality)
    
    features_dir = dataset_dir.joinpath("features")
    check_dir_exists(features_dir)
    
    output_dir = features_dir.joinpath(modality)
    check_dir_exists(output_dir)

    for video_dir in frames_dir.iterdir():
        output_video_dir = output_dir.joinpath(video_dir.name)
        check_dir_exists(output_video_dir)
        
        frame_files = os_sorted(str(v) for v in video_dir.rglob(glob_pattern))
        starts = list(range(0, len(frame_files) - window_length + 1, window_stride))
        
        extracted_count = len(list(output_video_dir.rglob("*.npy")))
        
        if extracted_count == len(starts):
            print(
                f"[INFO] Features for {video_dir.name} will not be extracted,\n",
                f"because the directory seems to already contain said features.\n",
                f"If you want to save the new features be sure to delete the old\n"
                f"ones before running a new extraction!"
            )
            continue
        
        input_data = []
        
        for file in frame_files:
            if modality == "rgb":
                frame = cv2.imread(file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame, dtype=np.float32)
                frame /= 255.0 # I forgot to normalize, so I have to extract the features again
                input_data.append(frame)
            else:
                flow = np.load(file)
                input_data.append(flow)
        
        input_data = np.array(input_data)
        
        for start in starts:
            print("[INFO] Extracting features.")
            print(f"Video: {video_dir.name}")
            print(f"Interval: {start} - {start + window_length}")
            
            input_data_batch = np.expand_dims(input_data[start : start + window_length], axis=0)
            features = np.squeeze(feature_extractor.predict(input_data_batch))
            
            print("Extracted features!")
            print(f"Feature shape: {features.shape}")
            output_feature_path = output_video_dir.joinpath(f"features_{start}_{start + window_length}.npy")
            
            if not output_feature_path.exists():
                np.save(output_feature_path, features)
                print(f"[INFO] Saving features to {output_feature_path}\n")
            else:
                print(f"[INFO] Not saving features, because they already exist!\n")
    
    
if __name__ == "__main__":
    task = "binary"
    modality = "rgb"
    epochs = 10
    model_name = f"C3D_{task}_{modality}_{epochs}_epochs"
    run_extraction("/home/elniad/datasets/boxing", model_name, window_length=8, window_stride=1)