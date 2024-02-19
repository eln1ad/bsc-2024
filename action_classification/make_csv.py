import pandas as pd
from pathlib import Path
from utils import check_dir_exists, check_file_exists


COLUMNS = ["video_name", "seg_start", "seg_end", "label"]


data_dir = Path.cwd().joinpath("data")
detection_data_dir = data_dir.joinpath("detection")
action_classification_data_dir = data_dir.joinpath("action_classification")


check_dir_exists(data_dir)
check_dir_exists(detection_data_dir)
check_dir_exists(action_classification_data_dir)


def make_csvs(train_pct = 0.8, rnd_seed = 63):
    for set_name in ["train", "val", "test"]:
        check_file_exists(detection_data_dir.joinpath(f"{set_name}.csv"))
    
    train_df = pd.read_csv(detection_data_dir.joinpath("train.csv"), index_col=None)
    val_df = pd.read_csv(detection_data_dir.joinpath("val.csv"), index_col=None)
    test_df = pd.read_csv(detection_data_dir.joinpath("test.csv"), index_col=None)
    
    df = pd.concat([train_df, val_df, test_df])
    
    # get actions
    df = df[df["label_str"] != "background"].reset_index(drop=True)
    
    # drop old columns
    df = df.drop(columns=["gt_start", "gt_end", "label", "tiou", "delta_center", "delta_length"])
    # rename old column
    df = df.rename(columns={"label_str": "label"})
    
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
            action_classification_data_dir.joinpath(f"{set_name}.csv"),
            index=None
        )
        
    print(
        f"[INFO] There will be {min_count} samples per label.\n"
        "[INFO] Made train, test, validation csv files!"
    )
    

if __name__ == "__main__":
    df = make_csvs()    