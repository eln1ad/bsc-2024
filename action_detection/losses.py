import tensorflow as tf
import numpy as np


EPSILON = 10e-8


def weigthed_binary_crossentropy(target_labels, pred_labels, w_positive = 1.0, w_negative = 1.0):
    """
    pred_labels: [Nx1],
    target_labels: [Nx1],
    w_positive: scalar,
    w_negative: scalar,
    
    By default the weights are equal. If there are much more negatives then positives,
    then the weights should be adjusted accordingly.
    """
    pos_log = tf.math.log(tf.clip_by_value(pred_labels, EPSILON, np.inf))
    neg_log = tf.math.log(tf.clip_by_value(1.0 - pred_labels, EPSILON, np.inf))
    return (-1.0) * tf.math.reduce_mean(w_positive * target_labels * pos_log + w_negative * (1.0 - target_labels) * neg_log)


def localization_loss(target_labels, target_delta_centers, target_delta_lengths, pred_delta_centers, pred_delta_lengths):
    """
    target_labels: [Nx1],
    target_centers: [Nx1],
    target_lengths: [Nx1],
    pred_labels: [Nx1],
    pred_centers: [Nx1],
    pred_lengths: [Nx1]
    
    Only those items will be added to the loss whose labels are equal to 1.0
    """
    # action_mask = tf.cast(tf.equal(target_labels, 1.0), dtype=tf.float32) # Nem kell átmaskolni, mert a labelek alapból 0-k és 1-k lehetnek csak
    
    loss_delta_centers = tf.math.reduce_mean(tf.square(target_delta_centers - pred_delta_centers) * target_labels)
    loss_delta_lengths = tf.math.reduce_mean(tf.square(target_delta_lengths - pred_delta_lengths) * target_labels)
    
    return loss_delta_centers + loss_delta_lengths