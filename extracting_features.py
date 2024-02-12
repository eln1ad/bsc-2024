import cv2
import numpy as np
from pathlib import Path
from natsort import os_sorted
from c3d_feature_extractor import build_feature_extractor


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
        paths = os_sorted(map(lambda x: str(x), video_dir.rglob(pattern)))
        
        items = []
        
        for path in paths:
            if modality == "rgb":
                x = cv2.imread(path)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                items.append(x)
            else:
                x = np.load(path)
                items.append(x)
        
        items = np.array(items)
        
        video_features_dir = features_dir.joinpath(video_dir.name)
        if not video_features_dir.exists():
            print(f"{video_dir} {modality} features directory did not exist. Creating it now!")
            video_features_dir.mkdir()
        
        for i in range(0, items.shape[0] - w_size + 1, w_stride):
            print("\nExtracting features...")
            print(f">>> Video: {video_dir.name}")
            print(f">>> Frames: {i} - {i + w_size}")
            img_batch = np.expand_dims(items[i : i + w_size], axis=0)
            img_features = np.squeeze(feature_extractor.predict(img_batch))
            print("Extracted features!")
            print(f">>> Shape: {img_features.shape}")
            img_features_save_path = video_features_dir.joinpath(f"features_{i}_{i+w_size}.npy")
            
            if not img_features_save_path.exists():
                np.save(img_features_save_path, img_features)
                print(f"Saving features to {img_features_save_path}\n")
            else:
                print(f"Not saving features, they already exist!\n")  
    
    
if __name__ == "__main__":
    main("/home/elniad/datasets/boxing", "C3D_rgb_8_frames_40_epochs", w_size=8, w_stride=1)