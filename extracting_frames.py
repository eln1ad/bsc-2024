import numpy as np
from pathlib import Path
from video import Video
import cv2


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
        
    modality = modality.lower()
        
    if modality not in VALID_MODALITIES:
        raise ValueError("Parameter 'modality' can only be rgb or flow")
        
    for video_path in video_path_list:
        print(f"Opening {video_path.name} for processing!")
        
        cap = cv2.VideoCapture(str(video_path))
        video_name = video_path.stem
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out_video_dir = Path(output_dir).joinpath(video_name)
        
        if out_video_dir.exists():
            # check if all frames were extracted
            if modality == "rgb":
                pattern = "*.png"
            else:
                pattern = "*.npy"
                num_frames -= 1 # optical flowból egyel kevesebb van, mint ahány frame
            
            frame_paths = list(out_video_dir.glob(pattern))
            
            if len(frame_paths) == num_frames:
                print(f"It seems that all frames were extracted from {video_path.name}!")
                print("Moving onto the next video!")
                continue
        else:
            print(f"{out_video_dir.name} directory does not exist, creating it now!")
            out_video_dir.mkdir()
        
        frames = []
        
        # read in every frame
        while cap.isOpened():
            ret, frame = cap.read() # BGR
            if not ret:
                break
            frames.append(frame)
            
        frames = [cv2.resize(frame, frame_size) for frame in frames]
        
        if modality == "rgb":
            for i, frame in enumerate(frames):
                out_frame_path = out_video_dir.joinpath(f"{i + 1}.png")
                cv2.imwrite(str(out_frame_path), frame) # will be saved as RGB
                print(f"Saved RGB image as {out_frame_path.name} to {out_video_dir}!")
        else:
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
            current_frame = gray_frames[0]

            for i, next_frame in enumerate(gray_frames[1:]):
                optical_flow = cv2.calcOpticalFlowFarneback(
                    current_frame,
                    next_frame,
                    None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                current_frame = next_frame
                out_flow_path = out_video_dir.joinpath(f"{i + 1}.npy")
                np.save(out_flow_path, optical_flow)
                print(f"Saved optical flow as {out_flow_path.name} to {out_video_dir}!")
                
        print(f"Finished processing {video_path.name}!")
        
    print("Finished processing all videos!")
                
                
if __name__ == "__main__":
    extract_frames(
        videos_dir="/home/elniad/datasets/boxing/videos",
        output_dir="/home/elniad/datasets/boxing/frames/rgb",
        frame_size=(112, 112),
        modality="rgb"
    )
    
    extract_frames(
        videos_dir="/home/elniad/datasets/boxing/videos",
        output_dir="/home/elniad/datasets/boxing/frames/flow",
        frame_size=(112, 112),
        modality="flow"
    )