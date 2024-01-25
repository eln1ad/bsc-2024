import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple


VALID_COLOR_FORMATS = ["rgb", "gray"]


class Video:
    def __init__(self, path):
        if not Path(path).exists():
            raise ValueError("File does not exist!")
        
        if Path(path).suffix != ".mp4":
            raise ValueError("File does not have an .mp4 extension!")
        
        self.path = path
        self.cap = cv2.VideoCapture(path)
        
        
    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))


    def read_frames(self, start=None, end=None, frame_size=None, color_format="rgb"):
        if color_format.lower() not in VALID_COLOR_FORMATS:
            raise ValueError(
                "Parameter 'color_format' can only take on\n"
                "the following values: rgb, gray"
            )
            
        if color_format == "rgb":
            cv2_color_format = cv2.COLOR_BGR2RGB
        else:
            cv2_color_format = cv2.COLOR_BGR2GRAY
        
        if frame_size is None:
            raise ValueError("Parameter 'frame_size' can not be empty!")
        
        if not isinstance(frame_size, tuple) or not isinstance(frame_size[0], int) or not isinstance(frame_size[1], int):
            raise ValueError("Parameter 'frame_size' must be a tuple[int, int]!")
        
        # autoset
        if start is None:
            start = 0
            
        if not isinstance(start, int):
            raise ValueError("Parameter 'start' must be an int!")
        
        if start > self.frame_count():
            raise ValueError("Parameter 'start' must be less than number of frames in the video!")
        
        if start < 0:
            raise ValueError("Parameter 'start' must be greater than or equal to 0!")
        
        # autoset
        if end is None:
            end = self.frame_count() - 1
            
        if not isinstance(end, int):
            raise ValueError("Parameter 'end' must be an int!")
        
        if end < 0:
            raise ValueError("Parameter 'end' must be greater than or equal to 0!")
        
        if end <= start:
            raise ValueError("Parameter 'end' must be greater than 'start'!")
        
        if end > self.frame_count():
            raise ValueError("Parameter 'end' must be less than or equal to the number of frames in the video!")
        
        video_name = Path(self.path).name
        print(f"Reading {color_format.upper()} frames from {video_name} at interval [{start},{end}] ...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
            
        while start < end + 1:
            grabbed, frame = self.cap.read()
            if not grabbed:
                break
            frame = cv2.cvtColor(frame, cv2_color_format)
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
            start += 1
            
        print(f"Finished reading {color_format.upper()} frames!")
        return np.array(frames)
    
    
    def read_optical_flows(self, start=None, end=None, frame_size=None):
        gray_frames = self.read_frames(start=start, end=end, frame_size=frame_size, color_format="gray")
        optical_flows = []
        current_frame = gray_frames[0]

        for next_frame in gray_frames[1:]:
            optical_flow = cv2.calcOpticalFlowFarneback(
                current_frame,
                next_frame,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            optical_flows.append(optical_flow)
            current_frame = next_frame
            
        print(f"Finished making {len(optical_flows)} FLOW frames from GRAY frames!")
        return np.array(optical_flows)

        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract RGB or Optical Flow frames from a given video."
    )
    
    parser.add_argument(
        "path",
        help="Path to the video, must be an .mp4 file!",
        type=str,
    )
    
    parser.add_argument(
        "-s",
        "--start",
        help="The index of the first frame to extract from the video.",
        default=None,
        type=int,
    )
    
    parser.add_argument(
        "-e",
        "--end",
        help="The index of the last frame to extract from the video.",
        default=None,
        type=int,
    )
    
    parser.add_argument(
        "-ff",
        "--frame_format",
        help="The format of the frames which are to be read, can be either RGB or FLOW",
        required=True,
        choices=["rgb", "flow"],
        type=str,
        dest="frame_format",
    )
    
    parser.add_argument(
        "-fs",
        "--frame_size",
        help="The size of the frames which are to be read, must be a tuple!",
        required=True,
        nargs="+",
        type=int,
        dest="frame_size"
    )
    
    parser.add_argument(
        "--output_dir",
        help="The directory where the frames should be saved as an .npy file",
        type=str,
    )
    
    return parser.parse_args()
    

if __name__ == "__main__":
    args = vars(parse_args())
    
    if len(args["frame_size"]) != 2:
        raise ValueError("'frame_size' can only take 2 arguments!")
    
    frame_size = tuple(args["frame_size"])
    video = Video(args["path"])
    
    if args["frame_format"] == "rgb":
        frames = video.read_frames(args["start"], args["end"], frame_size=frame_size, color_format="rgb")
    else:
        frames = video.read_optical_flows(args["start"], args["end"], frame_size=frame_size)
        
    print(f"Shape of frames {frames.shape}")
    
    if args["start"] is None:
        start = 0
        
    if args["end"] is None:
        end = video.frame_count()
    
    save_name = f"{Path(args['path']).stem}_{start}_{end}.npy"
    
    # the frame should be saved somewhere
    output_dir = Path(args["output_dir"])
    
    if not output_dir.exists():
        output_dir.mkdir()
        
    output_file = output_dir.joinpath(save_name)
    np.save(output_file, frames)
    print(f"Saved frames to {output_file}!")