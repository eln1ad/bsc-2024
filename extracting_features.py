import cv2
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from natsort import os_sorted


# Which layer should be used in C3D? relu_fc_6
def build_feature_extractor(model_name, last_layer_name="relu_fc_6"):
    saved_model_path = Path.cwd().joinpath("saved_models").joinpath(model_name)
    
    if not saved_model_path.exists():
        raise ValueError("Model with this name does not exist!")
    
    model = keras.models.load_model(saved_model_path)
    last_layer = model.get_layer(name=last_layer_name)
    model = tf.keras.Model(inputs=model.inputs, outputs=last_layer.output)
    return model


def main(dataset_dir, model_name, w_size, w_stride):
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
        
    feature_extractor = build_feature_extractor(model_name)
    
    if "rgb" in model_name:
        modality = "rgb"
        pattern = "*.png"
    elif "flow" in model_name:
        modality = "flow"
        pattern = "*.npy"
    else:
        raise ValueError
        
        
    frames_dir = dataset_dir.joinpath("frames", modality)
    features_dir = dataset_dir.joinpath("features", modality)
        
    if not features_dir.exists():
        print(f"{modality} features directory did not exist. Creating it now!")
        features_dir.mkdir(parents=True)

    for video_dir in frames_dir.iterdir():
        video_features_dir = features_dir.joinpath(video_dir.name)
        
        if not video_features_dir.exists():
            print(f"{video_dir} {modality} features directory did not exist. Creating it now!")
            video_features_dir.mkdir()
        
        frame_files = os_sorted(map(lambda x: str(x), video_dir.rglob(pattern)))
        steps = list(range(0, len(frame_files) - w_size + 1, w_stride))
        num_features = len(steps)
        num_features_extracted = len(list(video_features_dir.rglob("*.npy")))
        
        if num_features_extracted == num_features:
            print(f"All features have been extracted for {video_dir.name}!")
            continue
        
        X = []
        
        for file in frame_files:
            if modality == "rgb":
                frame = cv2.imread(file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame, dtype=np.float32)
                frame /= 255.0 # I forgot to normalize, so I have to extract the features again
                X.append(frame)
            else:
                flow = np.load(file)
                X.append(flow)
        
        X = np.array(X)
        
        for i in steps:
            print("\nExtracting features...")
            print(f">>> Video: {video_dir.name}")
            print(f">>> Frames: {i} - {i + w_size}")
            X_batch = np.expand_dims(X[i : i + w_size], axis=0)
            features = np.squeeze(feature_extractor.predict(X_batch))
            print("Extracted features!")
            print(f">>> Shape: {features.shape}")
            features_save_path = video_features_dir.joinpath(f"features_{i}_{i+w_size}.npy")
            
            if not features_save_path.exists():
                np.save(features_save_path, features)
                print(f"Saving features to {features_save_path}\n")
            else:
                print(f"Not saving features, they already exist!\n")
    
    
if __name__ == "__main__":
    main("/home/elniad/datasets/boxing", "C3D_rgb_8_frames_40_epochs", w_size=8, w_stride=1)