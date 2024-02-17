import json
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from video_classification.generator import get_classification_generator
from video_classification.models import C3D
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard


# 0. define paths and constants
TASK = "binary"


checkpoints_dir = Path.cwd().joinpath("checkpoints")
saved_models_dir = Path.cwd().joinpath("saved_models")
figures_dir = Path.cwd().joinpath("figures")
logs_dir = Path.cwd().joinpath("logs")
configs_dir = Path.cwd().joinpath("configs")
data_dir = Path.cwd().joinpath("data")
classification_data_dir = data_dir.joinpath("classification")


train_csv = classification_data_dir.joinpath(f"{TASK}_train.csv")
val_csv = classification_data_dir.joinpath(f"{TASK}_val.csv")


if not checkpoints_dir.exists():
    print("[INFO] checkpoints directory does not exist, creating it now!")
    checkpoints_dir.mkdir()
 
   
if not saved_models_dir.exists():
    print("[INFO] saved_models directory does not exist, creating it now!")
    saved_models_dir.mkdir()

    
if not figures_dir.exists():
    print("[INFO] figures directory does not exist, creating it now!")
    figures_dir.mkdir()


if not logs_dir.exists():
    print("[INFO] logs directory does not exist, creating it now!")
    logs_dir.mkdir()

 
if not train_csv.exists():
    raise ValueError(f"[INFO] data/classification/{TASK}_train.csv does not exist!")


if not val_csv.exists():
    raise ValueError("[INFO] data/classification/{TASK}_val.csv does not exist!")

    
with open(configs_dir.joinpath("C3D.json"), "r") as file:
    c3d_config = json.load(file)
    

with open(configs_dir.joinpath("general.json"), "r") as file:
    general_config = json.load(file)

 
if c3d_config["color-channels"] == 2:
    modality = "flow"
elif c3d_config["color-channels"] == 3:
    modality = "rgb"
else:
    raise ValueError("[ERROR] 'color-channels' can only take the following values: (2, 3)\n"
                     "modify the C3D.json config file accordingly!")


# Saved frames to SSD
frames_dir = Path(general_config["frames_dir"]).joinpath(modality)

model_version = f"C3D_{TASK}_{modality}_{c3d_config['epochs']}_epochs"
model_save_path = saved_models_dir.joinpath(model_version)


train_gen = get_classification_generator(
    train_csv, frames_dir,
    num_frames=c3d_config["capacity"], 
    shuffle=True,
    task=TASK
)


val_gen = get_classification_generator(
    val_csv, frames_dir,
    num_frames=c3d_config["capacity"], 
    shuffle=True,
    task=TASK
)


if TASK == "binary":
    output_shape = (1,)
else:
    output_shape = (c3d_config["num-classes"],)


output_signature = (
    tf.TensorSpec(shape=(c3d_config["capacity"], c3d_config["image-width"], c3d_config["image-height"], c3d_config["color-channels"]), dtype=tf.float32),
    tf.TensorSpec(shape=output_shape, dtype=tf.float32)
)


train_dset = tf.data.Dataset.from_generator(
    train_gen, 
    output_signature=output_signature
).batch(c3d_config["batch-size"], drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE) # caching and prefetching helps to alleviate the problem of highly input bound models


val_dset = tf.data.Dataset.from_generator(
    val_gen, 
    output_signature=output_signature
).batch(c3d_config["batch-size"], drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
  
      
if c3d_config["optimizer"] in ["SGD", "sgd"]:
    optimizer = SGD(learning_rate=c3d_config["learning-rate"])
elif c3d_config["optimizer"] in ["Adam", "adam"]:
    optimizer = Adam(learning_rate=c3d_config["learning-rate"])
else:
    raise ValueError(f"[ERROR] Optimizer can only be Adam or SGD, modify configs/C3D.json file!")
 

if TASK == "binary":
    loss = BinaryCrossentropy()
    metric = BinaryAccuracy()
else:
    loss = CategoricalCrossentropy()
    metric = CategoricalAccuracy()
    
    
model = C3D()
model.summary()


callbacks = [
    EarlyStopping(
        monitor="loss",
        patience=10
    ),
    ModelCheckpoint(
        str(checkpoints_dir.joinpath(model_version)),
        monitor="categorical_accuracy",
        save_best_only=True,
        save_weights_only=False
    ),
    CSVLogger(
        filename=logs_dir.joinpath(f"train_log_{model_version}.csv"),
        append=False,
    ),
    TensorBoard(
        log_dir=str(logs_dir.joinpath(f"tensor_board_{model_version}")),
        profile_batch=(10, 20) # profile batches batches between 10 and 20
    )
]


model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)


# Training was killed due to OUT OF MEMORY problem, I am going to limit
# the steps per epoch so it takes up less memory
history = model.fit(train_dset, validation_data=val_dset,
                    epochs=c3d_config["epochs"], callbacks=callbacks, workers=6)


tf.keras.models.save_model(model, model_save_path)
print(f"Saved {model_version} model to saved_models directory!")


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.savefig(figures_dir.joinpath(f"train_history_{model_version}.png"))