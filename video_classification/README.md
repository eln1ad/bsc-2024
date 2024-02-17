## Run the scripts in the following order

Be sure to run the scripts from the project's **root** directory!

1. `python video_classification/make_csv.py`
2. `python video_classification/train.py`
3. (optional) for feature extraction, run `python extract_features.py` located at the project root.

After training the video classifier you can use the trained model for feature extraction. The extracted features can then be used for action detection in the future.
