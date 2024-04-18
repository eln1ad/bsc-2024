from main import pipeline
from pathlib import Path
import PySimpleGUI as sg
import json
import pandas as pd


NMS_THRESHOLD = 0.1


def detection_pipeline_callback(window: sg.Window, video_path: str):
    if not video_path:
        sg.popup_error("Load in a video first!")
        return
    
    predictions = pipeline(Path(video_path), NMS_THRESHOLD)
    
    num_punches = len(predictions)
    punch_lengths = [int(prediction[1]) - int(prediction[0]) + 1 for prediction in predictions]
    
    min_punch_length = min(punch_lengths)
    max_punch_length = max(punch_lengths)
    avg_punch_length = 0
    
    for punch_length in punch_lengths:
        avg_punch_length += punch_length
    
    avg_punch_length /= num_punches
    
    window["-PRED_TABLE-"].update(values=predictions)
    window["-NUM_PUNCHES-"].update(num_punches)
    window["-AVG_LENGTH-"].update(avg_punch_length)
    window["-MIN_LENGTH-"].update(min_punch_length)
    window["-MAX_LENGTH-"].update(max_punch_length)
    
    return predictions
    
    
def load_gt_annotations_callback(window: sg.Window, video_path: str):
    # get gt annotations json
    dataset_file = Path.cwd().joinpath("data", "dataset.json")
    
    with open(dataset_file, "r") as file:
        dataset = json.load(file)
        
    if Path(video_path).stem not in dataset:
        sg.popup_error("Video is not the dataset!")
        return
    
    annotations = []
    
    for annotation in dataset[Path(video_path).stem]["actions"]:
        annotations.append([annotation["start"], annotation["end"], annotation["label"]])
    
    window["-GT_TABLE-"].update(values=annotations)


def export_csv_callback(predictions, video_path: str):
    if not video_path:
        sg.popup_error("Load in a video first!")
        return
    
    if not predictions:
        sg.popup_error("Evaluate the video first!")
        return
    
    df = pd.DataFrame(predictions, columns=["start", "end", "confidence", "label"])
    df.to_csv(f"result_for_{Path(video_path).stem}.csv", index=False)
    sg.popup(f"Exported results at {Path.cwd().joinpath(Path(video_path).stem)}.csv")