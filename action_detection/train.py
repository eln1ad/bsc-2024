import time
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from action_detection.models import detector
from action_detection.losses import localization_loss, weigthed_binary_crossentropy
from action_detection.generator import get_detection_generator
from utils import check_dir_exists, check_file_exists, load_json

from config_check import (
    get_modality,
    get_input_shape,
    get_linear_units,
    get_epochs,
    get_optimizer,
    get_batch_size,
    get_positive_weight,
    get_negative_weight,
    get_lambda_regression,
    get_dropout_rate,
)


checkpoints_dir = Path.cwd().joinpath("checkpoints")
saved_models_dir = Path.cwd().joinpath("saved_models")
figures_dir = Path.cwd().joinpath("figures")
logs_dir = Path.cwd().joinpath("logs")
configs_dir = Path.cwd().joinpath("configs")
data_dir = Path.cwd().joinpath("data")
detection_data_dir = data_dir.joinpath("detection")


check_dir_exists(checkpoints_dir)
check_dir_exists(saved_models_dir)
check_dir_exists(figures_dir)
check_dir_exists(logs_dir)
check_dir_exists(data_dir)
check_dir_exists(detection_data_dir)


train_csv = detection_data_dir.joinpath("train.csv")
val_csv = detection_data_dir.joinpath("val.csv")


check_file_exists(train_csv)
check_file_exists(val_csv)


detector_config_json = configs_dir.joinpath("detector.json")
general_config_json = configs_dir.joinpath("general.json")


check_file_exists(detector_config_json)
check_file_exists(general_config_json)


detector_config = load_json(detector_config_json)
general_config = load_json(general_config_json)


modality = get_modality(detector_config)
input_shape = get_input_shape(detector_config)
epochs = get_epochs(detector_config)
batch_size = get_batch_size(detector_config)
optimizer = get_optimizer(detector_config)
positive_weight = get_positive_weight(detector_config)
negative_weight = get_negative_weight(detector_config)
lambda_regression = get_lambda_regression(detector_config)


#features_dir = f"/home/elniad/datasets/boxing/features/{modality}"
features_dir = Path(general_config["features_dir"]).joinpath(modality)


model_version = f"detector_{modality}_{epochs}_epochs"
model_save_path = saved_models_dir.joinpath(model_version)


train_gen = get_detection_generator(
    train_csv, features_dir,
    shuffle=True
)


val_gen = get_detection_generator(
    val_csv, features_dir,
    shuffle=True
)


output_signature = (
    tf.TensorSpec(shape=input_shape, dtype=tf.float32),
    (
        tf.TensorSpec(shape=(1,), dtype=tf.float32), 
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
)


train_dset = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=output_signature
).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE) # caching and prefetching helps to alleviate the problem of highly input bound models


val_dset = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=output_signature
).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


model = detector(
    feat_dim=input_shape[0], 
    num_units=get_linear_units(detector_config), 
    dropout_rate=get_dropout_rate(detector_config)
)
model.summary()


train_losses = []
val_losses = []


# training loop
for epoch in range(epochs):
    start_time = time.time()
    
    train_epoch_losses = []
    train_epoch_classification_losses = []
    train_epoch_regression_losses = []
    
    val_epoch_losses = []
    val_epoch_classification_losses = []
    val_epoch_regression_losses = []
    
    print(f"[INFO] Epoch {epoch + 1}.")
    
    for batch_idx, (features, targets) in enumerate(train_dset):
        target_labels, target_delta_centers, target_delta_lengths = targets

        with tf.GradientTape() as tape:
            pred_labels, pred_delta_centers, pred_delta_lengths = model(features)
            
            classification_loss = weigthed_binary_crossentropy(
                target_labels, pred_labels,
                w_positive=positive_weight, w_negative=negative_weight
            )
            
            regression_loss = localization_loss(
                target_labels, target_delta_centers, target_delta_lengths,
                pred_delta_centers, pred_delta_lengths
            ) * lambda_regression # lambda_regression jelenleg 2.0
            
            loss = classification_loss + regression_loss 
            
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        train_epoch_losses.append(loss)
        train_epoch_classification_losses.append(classification_loss)
        train_epoch_regression_losses.append(regression_loss)
        
        if batch_idx % 10 == 0:
            print("<<< TRAIN >>>")
            print(f"Elapsed time: {round(time.time() - start_time, 4)}s")
            # print(f"Train loss: {train_loss / train_batch_counter}\n")
            print(f"Total loss: {tf.math.reduce_mean(train_epoch_losses)}")
            print(f"Classification loss: {tf.math.reduce_mean(train_epoch_classification_losses)}")
            print(f"Regression loss: {tf.math.reduce_mean(train_epoch_regression_losses)}\n")
    
    train_losses.append(tf.math.reduce_mean(train_epoch_losses))
    
    for batch_idx, (features, targets) in enumerate(val_dset):
        target_labels, target_delta_centers, target_delta_lengths = targets
        
        pred_labels, pred_delta_centers, pred_delta_lengths = model(features)
            
        classification_loss = weigthed_binary_crossentropy(
            target_labels, pred_labels, 
            w_positive=positive_weight, w_negative=negative_weight
        )
        
        regression_loss = localization_loss(
            target_labels, target_delta_centers, target_delta_lengths,
            pred_delta_centers, pred_delta_lengths
        ) * lambda_regression # lambda_regression jelenleg 2.0
        
        loss = classification_loss + regression_loss 
        
        val_epoch_losses.append(loss)
        val_epoch_classification_losses.append(classification_loss)
        val_epoch_regression_losses.append(regression_loss)
        
        if batch_idx % 10 == 0:
            print("<<< VALIDATION >>>")
            print(f"Elapsed time: {round(time.time() - start_time, 4)}s")
            print(f"Total loss: {tf.reduce_mean(val_epoch_losses)}")
            print(f"Classification loss: {tf.reduce_mean(val_epoch_classification_losses)}")
            print(f"Regression loss: {tf.reduce_mean(val_epoch_regression_losses)}\n")
    
    val_losses.append(tf.math.reduce_mean(val_epoch_losses))
    
print("Finished training!")
model.save(model_save_path)

plt.plot(list(range(epochs)), train_losses, color="blue", label="loss")
plt.plot(list(range(epochs)), val_losses, color="red", label="val_loss")
plt.legend(loc="upper right")
plt.savefig(figures_dir.joinpath(f"train_history_{model_version}.png"))