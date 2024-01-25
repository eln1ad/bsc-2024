## Video Classifier

How to train the video classifier?

Small snippets can be sampled from every video, with a window size of 8 or 16. Every snippet should have a label attached to it, based on it's IoU value with the closest ground truth instance. If the IoU is higher than 0.5 then the window will be marked as positive, else negative.

## Current Problems
[ ] video classifier loss does not improve
