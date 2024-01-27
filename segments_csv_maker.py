import numpy as np
import json
from video import Video
from pathlib import Path
from segments import tious
import pandas as pd


COLUMNS = ["video_name", "seg_start", "seg_end", "gt_start", "gt_end", "label", "tiou"]

data_dir = Path.cwd().joinpath("data")


def make_csvs(videos_dir, upper_tiou_threshold = 0.6, lower_tiou_threshold = 0.15,
             segment_size = 16, segment_stride = 1, train_pct = 0.8, rnd_seed = 42):
    dataset_path = data_dir.joinpath("dataset.json")
    
    if not Path(dataset_path).exists():
        raise ValueError("File dataset.json does not exist!")
    
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
        
    print("Loaded dataset!")
    
    if not Path(videos_dir).exists():
        raise ValueError("'videos_dir' does not exist!")
    
    video_path_list = list(Path(videos_dir).rglob("*.mp4"))
    
    if len(video_path_list) == 0:
        raise ValueError("'videos_dir' has no videos in it!")
    
    data_list = []

    for video_path in video_path_list:
        video_name = video_path.stem
        video_data = dataset[video_name]
        video_num_frames = video_data["frames"]
        
        if video_num_frames < segment_size:
            print(f"Skipping {video_path.name}, because it does not have enough frames!")
            continue
        
        video_gt_list = video_data["actions"]
        
        segment_starts = np.arange(0, video_num_frames - segment_size + 1, segment_stride)
        segments = np.stack([segment_starts, segment_starts + segment_size], axis=-1)
        
        video_gt_array = np.array([[video_gt["start"], video_gt["end"]] for video_gt in video_gt_list])
        
        tiou_array = tious(video_gt_array, segments)
        max_tious = np.max(tiou_array, axis=0)
        max_gt_indices = np.argmax(tiou_array, axis=0)
        
        for i, max_tiou in enumerate(max_tious):
            max_gt_idx = max_gt_indices[i]
            segment = segments[i]
            
            if max_tiou >= upper_tiou_threshold:
                video_label = video_gt_list[max_gt_idx]["label"]
                matched_gt = video_gt_array[max_gt_idx]
            elif max_tiou <= lower_tiou_threshold:
                video_label = "background"
                matched_gt = np.array([0, 0]) # dummy ground-truth
            else:
                continue
            
            data_list.append([video_name, segment[0], segment[1], matched_gt[0], matched_gt[1], video_label, max_tiou])

    df = pd.DataFrame(data_list, columns=COLUMNS)
    # df.to_csv(f"segments_size_{segment_size}_stride_{segment_stride}.csv", index=False)
    # print("Made csv!")
    groupby = df.groupby(by="label")
    min_count = int(groupby.count().min()[0]) # pick minimum amount of elements from each group

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
            data_dir.joinpath(f"{set_name}_segments_size_{segment_size}_stride_{segment_stride}_tiou_high_{upper_tiou_threshold}_tiou_low_{lower_tiou_threshold}.csv"), 
            index=None
        )
    
    print("Made csv files!")


if __name__ == "__main__":
    make_csvs(
        "/home/elniad/datasets/boxing/videos",
        segment_size=8,
        segment_stride=1,
    )