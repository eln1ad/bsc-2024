import time
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from action_detector import action_detector, localization_loss, weigthed_binary_crossentropy
from action_detection_generator import action_detection_generator
from keras.optimizers import SGD, Adam


checkpoints_dir = Path.cwd().joinpath("checkpoints")
saved_models_dir = Path.cwd().joinpath("saved_models")
figures_dir = Path.cwd().joinpath("figures")
logs_dir = Path.cwd().joinpath("logs")
data_dir = Path.cwd().joinpath("data")

train_csv = data_dir.joinpath("train_binary_segments_size_8_stride_1_tiou_high_0.5_tiou_low_0.15.csv")
val_csv = data_dir.joinpath("val_binary_segments_size_8_stride_1_tiou_high_0.5_tiou_low_0.15.csv")

if not checkpoints_dir.exists():
    print("checkpoints directory does not exist, creating it now!")
    checkpoints_dir.mkdir()

if not saved_models_dir.exists():
    saved_models_dir.mkdir()
    print("saved_models directory does not exist, creating it now!")

if not figures_dir.exists():
    figures_dir.mkdir()
    print("figures directory does not exist, creating it now!")

if not logs_dir.exists():
    logs_dir.mkdir()
    print("logs directory does not exist, creating it now!")

if not train_csv.exists():
    raise ValueError("The file 'train_csv' points to does not exist!")

if not val_csv.exists():
    raise ValueError("The file 'val_csv' points to does not exist!")

with open(data_dir.joinpath("action_detector_config.json"), "r") as file:
    config = json.load(file)

if config["color-channels"] == 2:
    modality = "flow"
elif config["color-channels"] == 3:
    modality = "rgb"
else:
    raise ValueError("'color-channels' must be either 2 or 3!")

video_features_dir = f"/home/elniad/datasets/boxing/features/{modality}"

model_version = f"action_detector_{modality}_window_size_{config['window_size']}_window_stride_{config['window_stride']}_epochs_{config['epochs']}"
model_save_path = saved_models_dir.joinpath(model_version)

train_gen = action_detection_generator(
    train_csv, video_features_dir,
    shuffle=True
)

val_gen = action_detection_generator(
    val_csv, video_features_dir,
    shuffle=True
)

output_signature = (
    tf.TensorSpec(shape=(config["feature-dim"],), dtype=tf.float32),
    (
        tf.TensorSpec(shape=(1,), dtype=tf.float32), 
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
)

train_dset = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=output_signature
).batch(config["batch-size"], drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE) # caching and prefetching helps to alleviate the problem of highly input bound models

val_dset = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=output_signature
).batch(config["batch-size"], drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

if config["optimizer"] in ["SGD", "sgd"]:
    optimizer = SGD(learning_rate=config["learning-rate"])
elif config["optimizer"] in ["Adam", "adam"]:
    optimizer = Adam(learning_rate=config["learning-rate"])
else:
    raise ValueError(f"Unknown optimizer inside config file!")

model = action_detector(feat_dim=config["feature-dim"], num_units=config["linear-units"], dropout_rate=config["dropout-rate"])
model.summary()

train_losses = []
val_losses = []

# training loop
for epoch in range(config["epochs"]):
    start_time = time.time()
    
    train_total_loss = 0
    train_classification_loss = 0
    train_regression_loss = 0
    train_batch_counter = 0
    
    val_total_loss = 0
    val_classification_loss = 0
    val_regression_loss = 0
    val_batch_counter = 0
    
    print(f"Epoch {epoch + 1}.")
    
    for batch_idx, (features, targets) in enumerate(train_dset):
        target_labels, target_delta_centers, target_delta_lengths = targets

        with tf.GradientTape() as tape:
            pred_labels, pred_delta_centers, pred_delta_lengths = model(features)
            
            classification_loss = weigthed_binary_crossentropy(
                target_labels, pred_labels,
                w_positive=config["w-positive"], w_negative=config["w-negative"]
            )
            
            regression_loss = localization_loss(
                target_labels, target_delta_centers, target_delta_lengths,
                pred_delta_centers, pred_delta_lengths
            ) * config["lambda-regression"] # Ez most 10.0-re van állítva, tehát ha a regressziónál kicsit hibázik, akkor az nagyon bele fog számítani a loss-ba
            
            batch_loss = classification_loss + regression_loss
            
        gradients = tape.gradient(batch_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        train_total_loss += batch_loss
        train_classification_loss += classification_loss
        train_regression_loss += regression_loss
        
        train_batch_counter += 1
        
        if batch_idx % 10 == 0:
            print(f"Elapsed time: {round(time.time() - start_time, 4)}s")
            # print(f"Train loss: {train_loss / train_batch_counter}\n")
            print(f"Total loss: {train_total_loss / train_batch_counter}")
            print(f"Classification loss: {train_classification_loss / train_batch_counter}")
            print(f"Regression loss: {train_regression_loss / train_batch_counter}\n")
    
    train_losses.append(train_total_loss / train_batch_counter)
    
    for batch_idx, (features, targets) in enumerate(val_dset):
        target_labels, target_delta_centers, target_delta_lengths = targets
        
        pred_labels, pred_delta_centers, pred_delta_lengths = model(features)
            
        classification_loss = weigthed_binary_crossentropy(
            target_labels, pred_labels, 
            w_positive=config["w-positive"], w_negative=config["w-negative"]
        )
        
        regression_loss = localization_loss(
            target_labels, target_delta_centers, target_delta_lengths,
            pred_delta_centers, pred_delta_lengths
        )
        
        val_total_loss += (regression_loss + classification_loss)
        val_classification_loss += classification_loss
        val_regression_loss += regression_loss
        
        val_batch_counter += 1
        
        if batch_idx % 10 == 0:
            print(f"Elapsed time: {round(time.time() - start_time, 4)}s")
            # print(f"Train loss: {train_loss / train_batch_counter}\n")
            print(f"Total loss: {val_total_loss / val_batch_counter}")
            print(f"Classification loss: {val_classification_loss / val_batch_counter}")
            print(f"Regression loss: {val_regression_loss / val_batch_counter}\n")
    
    val_losses.append(val_total_loss / val_batch_counter) 
    
print("Finished training!")
model.save(model_save_path)

plt.plot(list(range(config["epochs"])), train_losses, color="blue", label="train loss")
plt.plot(list(range(config["epochs"])), val_losses, color="red", label="val loss")
plt.legend()
plt.savefig(figures_dir.joinpath(f"training_history_{model_version}.png"))