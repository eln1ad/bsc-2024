import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from video_classification.generator import get_video_classification_generator
from video_classification.models import C3D
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard

from utils import check_dir_exists, check_file_exists, load_json
from config_check import (
    get_batch_size,
    get_dropout_rate,
    get_epochs,
    get_modality,
    get_optimizer,
    get_input_shape,
    get_output_shape,
    get_num_classes,
    get_task,
    get_linear_units,
)


checkpoints_dir = Path.cwd().joinpath("checkpoints")
saved_models_dir = Path.cwd().joinpath("saved_models")
figures_dir = Path.cwd().joinpath("figures")
logs_dir = Path.cwd().joinpath("logs")
configs_dir = Path.cwd().joinpath("configs")
data_dir = Path.cwd().joinpath("data")
video_classification_data_dir = data_dir.joinpath("video_classification")


c3d_json = configs_dir.joinpath("C3D.json")
paths_json = configs_dir.joinpath("paths.json")


check_dir_exists(checkpoints_dir)
check_dir_exists(saved_models_dir)
check_dir_exists(figures_dir)
check_dir_exists(logs_dir)


check_file_exists(c3d_json)
check_file_exists(paths_json)


model_config = load_json(c3d_json)
paths_config = load_json(paths_json)


modality = get_modality(model_config)
task = get_task(model_config)
epochs = get_epochs(model_config)
input_shape = get_input_shape(model_config)
output_shape = get_output_shape(model_config)
optimizer = get_optimizer(model_config)
batch_size = get_batch_size(model_config)


train_csv = video_classification_data_dir.joinpath(f"{task}_train.csv")
val_csv = video_classification_data_dir.joinpath(f"{task}_val.csv")


check_file_exists(train_csv)
check_file_exists(val_csv)


# Saved frames to SSD
frames_dir = Path(paths_config["frames_dir"]).joinpath(modality)


model_name = f"C3D_{task}_{modality}_{epochs}_epochs"
model_save_path = saved_models_dir.joinpath(model_name)


train_gen = get_video_classification_generator(
    train_csv, frames_dir,
    num_frames=input_shape[0], 
    shuffle=True,
    task=task
)


val_gen = get_video_classification_generator(
    val_csv, frames_dir,
    num_frames=input_shape[0], 
    shuffle=True,
    task=task
)


output_signature = (
    tf.TensorSpec(shape=input_shape, dtype=tf.float32),
    tf.TensorSpec(shape=output_shape, dtype=tf.float32)
)


train_dset = tf.data.Dataset.from_generator(
    train_gen, 
    output_signature=output_signature
).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE) # caching and prefetching helps to alleviate the problem of highly input bound models


val_dset = tf.data.Dataset.from_generator(
    val_gen, 
    output_signature=output_signature
).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
 

if task == "binary":
    loss = BinaryCrossentropy()
    metric = BinaryAccuracy()
else:
    loss = CategoricalCrossentropy()
    metric = CategoricalAccuracy()
    
    
model = C3D(
    input_shape=input_shape,
    dropout_pct=get_dropout_rate(model_config),
    num_classes=get_num_classes(model_config),
    linear_units=get_linear_units(model_config),
)
model.summary()

print(f"[INFO] Starting training {model_name}!")


callbacks = [
    EarlyStopping(
        monitor="loss",
        patience=10
    ),
    ModelCheckpoint(
        str(checkpoints_dir.joinpath(model_name)),
        monitor=("binary_accuracy" if task == "binary" else "categorical_accuracy"),
        save_best_only=True,
        save_weights_only=False
    ),
    CSVLogger(
        filename=logs_dir.joinpath(f"train_log_{model_name}.csv"),
        append=False,
    ),
    TensorBoard(
        log_dir=str(logs_dir.joinpath(f"tensor_board_{model_name}")),
        profile_batch=(10, 20) # profile batches batches between 10 and 20
    )
]


model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)


history = model.fit(train_dset, validation_data=val_dset, epochs=epochs, callbacks=callbacks)


tf.keras.models.save_model(model, model_save_path)
print(f"[INFO] Saved {model_name} model to saved_models directory!")


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig(figures_dir.joinpath(f"train_history_{model_name}.png"))


print(f"[INFO] Saved train history figure to figures directory!")