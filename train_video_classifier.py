import json
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from segments_generator import get_frames_label_generator
from c3d import c3d


EPOCHS = 50


c3d_checkpoint_dir = Path.cwd().joinpath("c3d_checkpoint")

if not c3d_checkpoint_dir.exists():
    print("'c3d_checkpoint_dir' does not exist, creating it now!")
    c3d_checkpoint_dir.mkdir()

with open("c3d_config.json", "r") as file:
    config = json.load(file)

train_csv_path = Path.cwd().joinpath("train_segments_size_8_stride_1.csv")
val_csv_path = Path.cwd().joinpath("val_segments_size_8_stride_1.csv")

if config["color-channels"] == 2:
    video_frames_dir = "/media/elniad/4tb_hdd/boxing-frames/flow"
elif config["color-channels"] == 3:
    video_frames_dir = "/media/elniad/4tb_hdd/boxing-frames/rgb"
else:
    raise ValueError("'color-channels' must be either 2 or 3!")

train_gen = get_frames_label_generator(train_csv_path, video_frames_dir, num_frames=config["capacity"], shuffle=True)
val_gen = get_frames_label_generator(val_csv_path, video_frames_dir, num_frames=config["capacity"],  shuffle=True)

output_signature = (
    tf.TensorSpec(shape=(config["capacity"], config["image-width"], config["image-height"], config["color-channels"]), dtype=tf.float32),
    tf.TensorSpec(shape=(config["num-classes"],), dtype=tf.float32)
)

train_dset = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature).batch(16, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_dset = tf.data.Dataset.from_generator(val_gen, output_signature=output_signature).batch(16, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

model = c3d()
model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=config["learning-rate"])
loss = tf.keras.losses.CategoricalCrossentropy()
metric = tf.keras.metrics.CategoricalAccuracy()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="categorical_accuracy", 
                                     patience=10),
    tf.keras.callbacks.ModelCheckpoint(c3d_checkpoint_dir,
                                       monitor="categorical_accuracy",
                                       save_best_only=True,
                                       save_weights_only=False)
]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)

print("Compiled model!")

history = model.fit(train_dset, validation_data=val_dset,
                    epochs=EPOCHS, callbacks=callbacks)

if config["color-channels"] == 2:
    modality = "flow"
else:
    modality = "rgb"
    
saved_models_dir = Path.cwd().joinpath("saved_models")

if not saved_models_dir.exists():
    saved_models_dir.mkdir()
    print("saved_models directory does not exist, creating it now!")

model_version = f"C3D_{modality}_{config['capacity']}_frames_{EPOCHS}_epochs"
model_save_path = saved_models_dir.joinpath(model_version)
tf.keras.models.save_model(model, model_save_path)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(f"training_history_{model_version}.png")

print(f"Saved C3D model to {model_save_path}!")