import time
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from action_detector import action_detector, total_loss
from features_generator import get_features_label_generator
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

train_gen = get_features_label_generator(
    train_csv, video_features_dir,
    shuffle=True
)

val_gen = get_features_label_generator(
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

model = action_detector(feat_dim=config["feature-dim"], num_units=config["linear-units"])
model.summary()

train_losses = []
val_losses = []

# training loop
for epoch in range(config["epochs"]):
    start_time = time.time()
    
    train_loss = 0
    val_loss = 0
    train_batch_counter = 0
    val_batch_counter = 0
    
    print(f"Epoch {epoch + 1}.")
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_dset):
        with tf.GradientTape() as tape:
            pred_labels, pred_centers, pred_lengths = model(X_batch)
            
            train_loss += total_loss(
                y_batch,
                (pred_labels, pred_centers, pred_lengths),
                config["lambda-regression"],
                config["w-positive"],
                config["w-negative"],
            )
            
        gradients = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        train_batch_counter += 1
        
        if batch_idx % 10 == 0:
            print(f"Elapsed time: {round(time.time() - start_time, 4)}s")
            print(f"Train loss: {train_loss / train_batch_counter}\n")
    
    train_losses.append(train_loss / train_batch_counter)
    
    for batch_idx, (X_batch, y_batch) in enumerate(val_dset):
        pred_labels, pred_centers, pred_lengths = model(X_batch)
        val_loss += total_loss(
                y_batch,
                (pred_labels, pred_centers, pred_lengths),
                config["lambda-regression"],
                config["w-positive"],
                config["w-negative"],
            )
        
        val_batch_counter += 1
        
        if batch_idx % 10 == 0:
            print(f"Elapsed time: {round(time.time() - start_time, 4)}s")
            print(f"Val loss: {val_loss / val_batch_counter}\n")
        
    val_losses.append(val_loss / val_batch_counter)  
    
print("Finished training!")
model.save(model_save_path)

plt.plot(list(range(config["epochs"])), train_losses, color="blue", label="train loss")
plt.plot(list(range(config["epochs"])), val_losses, color="red", label="val loss")
plt.legend()
plt.savefig(figures_dir.joinpath(f"training_history_{model_version}.png"))