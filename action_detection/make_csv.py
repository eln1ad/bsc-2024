import numpy as np
import json
from pathlib import Path
from segments import tious
import pandas as pd
from utils import (
    load_json
)


COLUMNS = ["video_name", "seg_start", "seg_end", "gt_start", "gt_end", "label", "label_str", "tiou", "delta_center", "delta_length"]


data_dir = Path.cwd().joinpath("data")
action_detection_data_dir = data_dir.joinpath("action_detection")


def get_data_list(videos_dir, upper_tiou_threshold = 0.7, lower_tiou_threshold = 0.3,
             segment_sizes = [8, 10, 12, 14, 16], segment_overlap = 0.8):
    dataset_path = data_dir.joinpath("dataset.json")
    
    if not Path(dataset_path).exists():
        raise ValueError("[ERROR] data/dataset.jon does not exist!")
    
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
        
    if not Path(videos_dir).exists():
        raise ValueError("[ERROR] 'videos_dir' does not exist!")
    
    video_path_list = list(Path(videos_dir).rglob("*.mp4"))
    
    if len(video_path_list) == 0:
        raise ValueError("[ERROR] 'videos_dir' has no videos in it!")
    
    data_list = []
    
    for video_path in video_path_list:
        video_name = video_path.stem
        video_data = dataset[video_name]
        
        video_num_frames = video_data["frames"]
        
        for segment_size in segment_sizes:
            if (1 - segment_overlap) == 0:
                segment_stride = 1
            else:
                segment_stride = round(segment_size * (1 - segment_overlap))
                
            if video_num_frames < segment_size:
                print(
                    f"[INFO] {video_name} will not be used, during the generation of\n"
                    f"segments with size {segment_size} and stride {segment_stride}\n"
                    f"because it does not have enough frames!"
                )
                continue
                
            video_ground_truths = np.array([[x["start"], x["end"]] for x in video_data["actions"]])
            video_ground_truth_labels = np.array([x["label"] for x in video_data["actions"]])
            
            segment_starts = np.arange(0, video_num_frames - segment_size + 1, segment_stride)
            segment_ends = segment_starts + segment_size
            segments = np.stack([segment_starts, segment_ends], axis=-1)
            
            tiou_matrix = tious(video_ground_truths, segments)
            
            max_tious = np.max(tiou_matrix, axis=0)
            max_ground_truth_indices = np.argmax(tiou_matrix, axis=0)
            
            for i, max_tiou in enumerate(max_tious):
                max_idx = max_ground_truth_indices[i]
                segment = segments[i]
                
                if max_tiou >= upper_tiou_threshold:
                    label = 1.0 # action
                    label_str = video_ground_truth_labels[max_idx]
                    
                    ground_truth = video_ground_truths[max_idx]
                    ground_truth_length = ground_truth[1] - ground_truth[0]
                    ground_truth_center = (ground_truth[1] + ground_truth[0]) / 2.0
                    
                    segment_center = (segment[1] + segment[0]) / 2.0
                    
                    delta_center_norm = (ground_truth_center - segment_center) / segment_size
                    delta_length_norm = np.log(ground_truth_length / segment_size)
                    
                    data_list.append([
                        video_name, 
                        segment[0], segment[1], 
                        ground_truth[0], ground_truth[1],
                        label,
                        label_str,
                        max_tiou,
                        delta_center_norm,
                        delta_length_norm
                    ])
                elif max_tiou <= lower_tiou_threshold:
                    label = 0.0 # background
                    label_str = "background"
                    data_list.append([
                        video_name, 
                        segment[0], segment[1], 
                        0.0, 0.0,
                        label,
                        label_str,
                        max_tiou,
                        0.0, # delta center and length norms will be 0
                        0.0
                    ])
                else:
                    continue # will not be used
                
    return data_list
                

def make_csvs(data_list, train_pct = 0.8, rnd_seed = 63):
    df = pd.DataFrame(data_list, columns=COLUMNS)
    groupby = df.groupby(by="label")
    min_count = int(groupby.count().min()[0])
    
    train_count = round(min_count * train_pct)
    test_count = (min_count - train_count) // 2
                
    full_set = {
        "train": [],
        "test": [],
        "val": []
    }    
    
    for _, items in groupby:
        items = items.sample(n=min_count, random_state=rnd_seed)
        
        full_set["train"].extend(items.iloc[ : train_count].values.tolist())
        full_set["test"].extend(items.iloc[train_count : train_count + test_count].values.tolist())
        full_set["val"].extend(items.iloc[train_count + test_count :].values.tolist())
        
    for set_name, one_set in full_set.items():
        pd.DataFrame(one_set, columns=COLUMNS).to_csv(
            action_detection_data_dir.joinpath(f"{set_name}.csv"),
            index=None
        )
        
    print("[INFO] Made train, test, validation csv files!")


if __name__ == "__main__":
    configs_dir = Path.cwd().joinpath("configs")
    paths_config = load_json(configs_dir.joinpath("paths.json"))
    
    data_list = get_data_list(paths_config["videos_dir"], upper_tiou_threshold=0.7, lower_tiou_threshold=0.3,
                              segment_sizes=[8, 10, 12, 14, 16], segment_overlap=0.9)
    make_csvs(data_list, train_pct=0.9)
 