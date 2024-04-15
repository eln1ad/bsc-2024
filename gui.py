import PySimpleGUI as sg
import pandas as pd
import numpy as np
import cv2
from pathlib import Path


# keys
# -PRED_TABLE-
# -GT_TABLE-
# -TIME-
# -LOAD-
# -EVAL-
# -PLAY-
# -RESTART-
# -PRED_EXPORT-
# -NUM_PUNCHES-
# -AVG_LENGTH-
# -MIN_LENGTH-
# -MAX_LENGTH-
# -IMAGE-


F_WIDTH = 400
F_HEIGHT = 400


def encode_as_bytes(frame):
    frame = cv2.resize(frame, (F_WIDTH, F_HEIGHT))
    return cv2.imencode(".ppm", frame)[1].tobytes()

def update_image(window, frame):
    fbytes = encode_as_bytes(frame)
    window["-IMAGE-"].update(data=fbytes)


def load_video_from_beginning(window, cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Video has no frames!")
    update_image(window, frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def update_play_button(window, is_playing):
    window["-PLAY-"].update("Pause" if is_playing else "Play")
    
    
def update_time_text(window, cap):
    window["-TIME-"].update(f"Time: {round(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS), 4)} s\tFrame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
    


def main():
    # dummy data
    predictions = [[10, 22], [27, 29]]
    ground_truths = [[11, 22], [24, 29]]
    
    cap = None
    video = None
    is_playing = False
    
    left_column = sg.Column(
        [
            [
                sg.Column([
                    [sg.Text("Predictions")],
                    [sg.Table(
                        predictions, ["Start", "End"], num_rows=15, 
                        select_mode=sg.TABLE_SELECT_MODE_NONE, enable_events=False,
                        col_widths=[5, 5], expand_x=True, key="-PRED_TABLE-",
                    )]
                ]),
                sg.Column([
                    [sg.Text("Ground truths")],
                    [sg.Table(
                        ground_truths, ["Start", "End"], num_rows=15,
                        select_mode=sg.TABLE_SELECT_MODE_NONE, enable_events=False,
                        col_widths=[5, 5], expand_x=True, key="-GT_TABLE-",
                    )]
                ]),
                sg.Column([
                    [sg.Text("Time: 0s\tFrame: 0", key="-TIME-")],
                    [sg.FileBrowse("Load video", file_types=(("MP4 files", "*.mp4"),), enable_events=True, key="-LOAD-")],
                    [sg.Button("Evaluate video", key="-EVAL-")],
                    [sg.Button("Play", key="-PLAY-"), sg.Button("Restart", key="-RESTART-")],
                    [sg.Button("Export predictions as CSV", "-PRED_EXPORT-")],
                ], expand_x=True)
            ],
            [
                sg.Column([[sg.Text("Num punches:")], [sg.Text("", key="-NUM_PUNCHES-")]]),
                sg.Column([[sg.Text("Avg length:")], [sg.Text("", key="-AVG_LENGTH-")]]),
                sg.Column([[sg.Text("Min length:")], [sg.Text("", key="-MIN_LENGTH-")]]),
                sg.Column([[sg.Text("Max length:")], [sg.Text("", key="-MAX_LENGTH-")]]),
            ]
        ],
        expand_x=True
    )
    
    right_column = sg.Column(
        [
            [sg.Image(size=(F_WIDTH, F_HEIGHT), background_color="white", key="-IMAGE-")]
        ]
    )
    
    layout = [[left_column, right_column]]
    
    window = sg.Window("Boxing App", layout=layout, finalize=True)
    
    while True:
        event, values = window.read(timeout=50)
        
        if event == sg.WINDOW_CLOSED:
            break
        
        # user pressed load button
        if event == "-LOAD-":
            video_path = Path(values["-LOAD-"])
            video = video_path.name
            cap = cv2.VideoCapture(str(video_path))
            load_video_from_beginning(window, cap)
            
            # fill -GT_TABLE- with ground truth annotations
            
        if event == "-PLAY-":
            is_playing = not is_playing
            update_play_button(window, is_playing)
            
        if event == "-RESTART-":
            load_video_from_beginning(window, cap)
            update_time_text(window, cap)
            is_playing = False
            update_play_button(window, is_playing)
        
        if is_playing:
            update_time_text(window, cap)
            ret, frame = cap.read()
            if not ret:
                sg.popup("No more frames left!")
                is_playing = False
                continue
            update_image(window, frame)
            
            
            

    window.close()


if __name__ == "__main__":
    main()