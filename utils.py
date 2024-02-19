import json
import numpy as np
from pathlib import Path
from typing import List


def check_dir_exists(d: Path):
    if not d.exists():
        print(f"[INFO] {d.name} directory does not exist, creating it now!")
        d.mkdir()
        

def check_file_exists(f: Path):
    if not f.exists():
        raise ValueError(f"[INFO] {f.name} does not exist!")
        
        
def load_json(f: Path):
    check_file_exists(f)
    with open(f, "r") as file:
        return json.load(file)
    
    
def one_hot_encode(label_list: List[str], label: str):
    one_hot = np.zeros(len(label_list))
    one_hot[label_list.index(label)] = 1.0
    return one_hot


def check_norm_method(norm_method):
    if norm_method not in ["l2", "mean_std", "min_max", None]:
        raise ValueError("[ERROR] 'norm_method' can only be 'l2', 'mean_std', 'min_max' or None!")
    
    
def check_modality(modality):
    if modality not in ["rgb", "flow"]:
        raise ValueError("[Error] 'modality' can only be 'rgb' or 'flow'!")