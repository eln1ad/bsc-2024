import time
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from action_classification.models import classifier
from action_classification.generator import get_classification_generator

from config_check import (
    get_batch_size,
    get_dropout_rate,
    get_input_shape,
    get_linear_units,
    get_optimizer,
    get_modality,
    get_epochs
)

from utils import (
    check_dir_exists, 
    check_file_exists, 
    load_json,
)


checkpoints_dir = Path.cwd().joinpath("checkpoints")
saved_models_dir = Path.cwd().joinpath("saved_models")
figures_dir = Path.cwd().joinpath("figures")
logs_dir = Path.cwd().joinpath("logs")
configs_dir = Path.cwd().joinpath("configs")
data_dir = Path.cwd().joinpath("data")
action_classification_data_dir = data_dir.joinpath("action_classification")


check_dir_exists(checkpoints_dir)
check_dir_exists(saved_models_dir)
check_dir_exists(figures_dir)
check_dir_exists(logs_dir)
check_dir_exists(data_dir)
check_dir_exists(action_classification_data_dir)


train_csv = action_classification_data_dir.joinpath("train.csv")
val_csv = action_classification_data_dir.joinpath("val.csv")


check_file_exists(train_csv)
check_file_exists(val_csv)


paths_json = configs_dir.joinpath("paths.json")
check_file_exists(paths_json)
paths_config = load_json(paths_json)


action_classifier_json = configs_dir.joinpath("action_classifier.json")
check_file_exists(action_classifier_json)
model_config = load_json(action_classifier_json)


label_list_txt = action_classification_data_dir.joinpath("label_list.txt")
check_file_exists(label_list_txt)


modality = get_modality(model_config)
input_shape = get_input_shape(model_config)
epochs = get_epochs(model_config)
batch_size = get_batch_size(model_config)
optimizer = get_optimizer(model_config)


features_dir = Path(paths_config["features_dir"]).joinpath(modality)

model_name = f"action_classifier_{modality}_{epochs}_epochs"
model_save_path = saved_models_dir.joinpath(model_name)


train_gen = get_classification_generator(
    train_csv, label_list_txt,
    features_dir,
    shuffle=True
)


val_gen = get_classification_generator(
    val_csv, label_list_txt,
    features_dir,
    shuffle=True
)


label_list = []


with open(label_list_txt, "r") as file:
    for line in file.readlines():
        label = line.strip()
        label_list.append(label)

output_shape = (len(label_list),)


output_signature = (
    tf.TensorSpec(shape=input_shape, dtype=tf.float32),
    tf.TensorSpec(shape=output_shape, dtype=tf.float32),
)


train_dset = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=output_signature
).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE) # caching and prefetching helps to alleviate the problem of highly input bound models


val_dset = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=output_signature
).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


model = classifier(
    feat_dim=input_shape[0],
    num_units=get_linear_units(model_config),
    dropout_rate=get_dropout_rate(model_config)
)
model.summary()

loss = CategoricalCrossentropy()
metric = CategoricalAccuracy()

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric],
)


history = model.fit(train_dset, validation_data=val_dset, epochs=epochs)

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