import numpy as np


def intersections(segs_a, segs_b):
    return np.maximum(0.0, np.minimum(segs_b[:, 1], segs_a[:, 1]) - np.maximum(segs_b[:, 0], segs_a[:, 0])) + 1.0


def unions(segs_a, segs_b):
    lengths_a = segs_a[:, 1] - segs_a[:, 0] + 1.0
    lengths_b = segs_b[:, 1] - segs_b[:, 0] + 1.0
    
    return np.maximum(
        lengths_a + lengths_b - intersections(segs_a, segs_b),
        np.maximum(segs_b[:, 1], segs_a[:, 1]) - np.minimum(segs_b[:, 0], segs_a[:, 0]) + 1.0
    )
    

def tious(segs_gt, segs_target, flatten=False):
    num_gt = len(segs_gt)
    num_target = len(segs_target)
    
    segs_gt = np.tile(segs_gt, reps=[num_target, 1])
    segs_target = np.repeat(segs_target, repeats=num_gt, axis=0)
    
    tiou_vec = intersections(segs_gt, segs_target) / unions(segs_gt, segs_target)
    
    if not flatten:
        return np.transpose(np.reshape(tiou_vec, newshape=[num_target, num_gt]))

    return tiou_vec
        

if __name__ == "__main__":
    segs_gt = np.array([
        [10, 14],
        [15, 23],
        [78, 89],
    ])
    
    segs_target = np.array([
        [22, 29],
        [30, 34],
        [65, 78],
        [78, 90]
    ])
    
    print(tious(segs_gt, segs_target))
