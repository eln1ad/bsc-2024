import cv2
import numpy as np
from pathlib import Path
from video import Video


VALID_MODALITIES = ["rgb", "flow"]


def extract_frames(videos_dir, output_dir, frame_size = None, modality = "rgb"):
    if not Path(videos_dir).exists():
        raise ValueError("'videos_dir' does not exist!")
    
    video_path_list = list(Path(videos_dir).rglob("*.mp4"))
    
    if len(video_path_list) == 0:
        raise ValueError("'videos_dir' has no videos in it!")
    
    if not Path(output_dir).exists():
        print("'output_dir' does not exist, creating it now!")
        Path(output_dir).mkdir(parents=True)
        
    if modality.lower() not in VALID_MODALITIES:
        raise ValueError("Parameter 'modality' can only be rgb or flow")
        
    for video_path in video_path_list:
        video = Video(str(video_path))
        video_name = video_path.stem
        
        if modality.lower() == "rgb":
            frames = video.read_frames(frame_size=frame_size, color_format="rgb")
        else:
            frames = video.read_optical_flows(frame_size=frame_size)
            
        output_video_dir = Path(output_dir).joinpath(video_name)
        
        if not output_video_dir.exists():
            print(f"{output_video_dir} does not exist, creating it now!")
            output_video_dir.mkdir()
            
        for i, frame in enumerate(frames):
            frame_path = output_video_dir.joinpath(f"{i}.npy")
            
            if not frame_path.exists():
                np.save(frame_path, frame)
                print(f"Saved frame to {frame_path}!")
            else:
                print(
                    "Not saving frame, because there already exists a frame\n"
                    f"with the name {frame_path.name} inside the folder {output_video_dir.name}!")
                
        print(f"Finished saving frames for {video_path.name}!")
                
                
if __name__ == "__main__":
    extract_frames(
        videos_dir="/home/elniad/datasets/boxing/videos",
        output_dir="/media/elniad/4tb_hdd/datasets/boxing/frames/rgb",
        frame_size=(112, 112),
        modality="rgb"
    )
    
    extract_frames(
        videos_dir="/home/elniad/datasets/boxing/videos",
        output_dir="/media/elniad/4tb_hdd/datasets/boxing/frames/flow",
        frame_size=(112, 112),
        modality="flow"
    )