import tensorflow as tf
import numpy as np
from keras import Model, Input
from keras.layers import ReLU, Dense
from keras.activations import sigmoid


EPSILON = 10e-8


# input: (N, 1024) batch of feature vectors
# output: (N, 1) confidence scores, (N, 2) locations
# 2 intermediate layers + 1 top layer
def action_detector(feat_dim=1024, num_units=1024):
    input_layer = Input(shape=(feat_dim,))
    
    fc_1 = Dense(num_units)(input_layer)
    relu_1 = ReLU()(fc_1)
    fc_2 = Dense(num_units)(relu_1)
    relu_2 = ReLU()(fc_2)
    
    out_confidence = Dense(1)(relu_2)
    out_confidence = sigmoid(out_confidence)
    
    # location won't have an activation function, since it is linear
    out_center = Dense(1)(relu_2)
    out_length = Dense(1)(relu_2)
    
    model = Model(inputs=input_layer, outputs=[out_confidence, out_center, out_length])
    return model


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


def localization_loss(target_labels, target_centers, target_lengths, pred_labels, pred_centers, pred_lengths, lambda_regression):
    """
    target_labels: [Nx1],
    target_centers: [Nx1],
    target_lengths: [Nx1],
    pred_labels: [Nx1],
    pred_centers: [Nx1],
    pred_lengths: [Nx1]
    
    Only those items will be added to the loss whose labels are equal to 1.0
    """
    action_mask = tf.cast(tf.equal(target_labels, 1.0), dtype=tf.float32)
    
    loss_centers = tf.math.reduce_mean(tf.square(target_centers - pred_centers) * action_mask)
    loss_lengths = tf.math.reduce_mean(tf.square(target_lengths - pred_lengths) * action_mask)
    
    return (loss_centers + loss_lengths) * lambda_regression


def total_loss(targets, predictions, lambda_regression = 1.0, w_positive = 1.0, w_negative = 1.0):
    target_labels, target_centers, target_lengths = targets
    pred_labels, pred_centers, pred_lengths = predictions
    return (
        localization_loss(target_labels, target_centers, target_lengths, pred_labels, pred_centers, pred_lengths, lambda_regression) + 
        weigthed_binary_crossentropy(target_labels, pred_labels, w_positive=w_positive, w_negative=w_negative)
    )


if __name__ == "__main__":
    model = action_detector()
    model.summary()