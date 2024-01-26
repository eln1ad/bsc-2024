import json
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from segments_generator import get_frames_label_generator
from c3d import c3d
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau


checkpoints_dir = Path.cwd().joinpath("checkpoints")
saved_models_dir = Path.cwd().joinpath("saved_models")
figures_dir = Path.cwd().joinpath("figures")
logs_dir = Path.cwd().joinpath("logs")

train_csv = Path.cwd().joinpath("train_segments_size_8_stride_1.csv")
val_csv = Path.cwd().joinpath("val_segments_size_8_stride_1.csv")

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
    
with open("c3d_config.json", "r") as file:
    config = json.load(file)
    
if config["color-channels"] == 2:
    modality = "flow"
elif config["color-channels"] == 3:
    modality = "rgb"
else:
    raise ValueError("'color-channels' must be either 2 or 3!")

video_frames_dir = f"/media/elniad/4tb_hdd/datasets/boxing/frames/{modality}"

model_version = f"C3D_{modality}_{config['capacity']}_frames_{config['epochs']}_epochs"
model_save_path = saved_models_dir.joinpath(model_version)

# gpus = tf.config.list_physical_devices("GPU")

# if len(gpus):
#     print("Found GPU!")
#     print(gpus)
    
#     # ezt amiatt raktam ide, hogy egy warningot eltüntessek
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

train_gen = get_frames_label_generator(
    train_csv, video_frames_dir, 
    num_frames=config["capacity"], 
    shuffle=True
)

val_gen = get_frames_label_generator(
    val_csv, video_frames_dir, 
    num_frames=config["capacity"], 
    shuffle=True
)

output_signature = (
    tf.TensorSpec(shape=(config["capacity"], config["image-width"], config["image-height"], config["color-channels"]), dtype=tf.float32),
    tf.TensorSpec(shape=(config["num-classes"],), dtype=tf.float32)
)

train_dset = tf.data.Dataset.from_generator(
    train_gen, 
    output_signature=output_signature
).batch(config["batch-size"], drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_dset = tf.data.Dataset.from_generator(
    val_gen, 
    output_signature=output_signature
).batch(config["batch-size"], drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
if config["optimizer"] in ["SGD", "sgd"]:
    optimizer = SGD(learning_rate=config["learning-rate"])
elif config["optimizer"] in ["Adam", "adam"]:
    optimizer = Adam(learning_rate=config["learning-rate"])
else:
    raise ValueError(f"Unknown optimizer inside config file!")   

model = c3d()
model.summary()
    
loss = CategoricalCrossentropy()
metric = CategoricalAccuracy()

callbacks = [
    EarlyStopping(
        monitor="loss",
        patience=10
    ),
    ModelCheckpoint(
        str(checkpoints_dir.joinpath(f"checkpoint_{model_version}")),
        monitor="categorical_accuracy",
        save_best_only=True,
        save_weights_only=False
    ),
    CSVLogger(
        filename=logs_dir.joinpath(f"training_log_{model_version}.csv"),
        append=False,
    ),
    # ReduceLROnPlateau(
    #     monitor="loss",
    #     factor=0.1,
    #     min_lr=1e-5,
    #     patience=5, # this might need to be higher
    # )
]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)

# forcing tensorflow to train using gpu
# try these versions:
# /job:localhost/replica:0/task:0/device:GPU:0
# /GPU:0
# with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
history = model.fit(train_dset, validation_data=val_dset,
                    epochs=config["epochs"], callbacks=callbacks)

tf.keras.models.save_model(model, model_save_path)
print(f"Saved C3D model to {model_save_path}!")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.savefig(figures_dir.joinpath(f"training_history_{model_version}.png"))