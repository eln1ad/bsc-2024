import json
from pathlib import Path


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