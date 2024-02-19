from keras.optimizers import SGD, Adam
    
    
def get_modality(config):
    if config["modality"] not in ["rgb", "flow"]:
        raise ValueError("[ERROR] 'modality' can only be 'rgb' or 'flow', modify config!")
    return config["modality"]


def get_color_channels(config):
    modality = get_modality(config)
    if modality == "rgb":
        return 3
    if modality == "flow":
        return 2

   
def get_task(config):
    if config["task"] not in ["binary", "multi"]:
        raise ValueError("[ERROR] 'task' can only be 'binary' or 'multi', modify the config file!")
    return config["task"]


def get_num_classes(config):
    task = get_task(config)
    if config["num_classes"] <= 0:
        raise ValueError("[ERROR] 'num_classes' must be greater than 0, modify config!")
    if task == "binary":
        return 2
    return config["num-classes"] # last option is that task is 'multi'


def get_input_shape(config):
    l = config["capacity"]
    w = config["image_width"]
    h = config["image_height"]
    c = get_color_channels(config)
    
    if l < 8 or l > 16:
        raise ValueError("[ERROR] 'capacity' must be between 8 and 16, modify config!")
    if (w < 96 or w > 224) and (h < 96 or h > 224) and w != h:
        raise ValueError(
            "[ERROR] use spatial dimensions with shape (N, N), where\n"
            "N is >= 96 and <= 224, modify config!"
        )
    
    return (l, w, h, c)


def get_output_shape(config):
    if get_task(config) == "binary":
        return (1,)
    else:
        return (get_num_classes(config),)


def get_epochs(c3d_config):
    if c3d_config["epochs"] < 0:
        raise ValueError("[ERROR] 'epochs' must be greater than 0, modify config!")
    return c3d_config["epochs"]


def get_learning_rate(config):
    if config["learning_rate"] <= 0 and config["learning_rate"] > 0.5:
        raise ValueError("[ERROR] 'learning_rate' must be in the range of ]0, 0.5], modify config!")
    return config["learning_rate"]


def get_optimizer(config):
    learning_rate = get_learning_rate(config)
    optimizer = config["optimizer"].lower()
    
    if optimizer not in ["adam", "sgd"]:
        raise ValueError(f"[ERROR] 'optimizer' must be 'adam' or 'sgd', modify config!")
    if optimizer == "adam":
        return Adam(learning_rate=learning_rate)
    if optimizer == "sgd":
        return SGD(learning_rate=learning_rate)
    
    
def get_batch_size(config):
    if config["batch_size"] <= 0:
        raise ValueError("[ERROR] 'batch_size' must be greater than 0, modify config!")
    if not config["batch_size"] % 2 == 0:
        raise ValueError("[ERROR] 'batch_size' must be divisable by 2, modify config!")
    return config["batch_size"]


def get_dropout_rate(config):
    dropout_rate = config["dropout_rate"]
    if not isinstance(dropout_rate, float):
        raise ValueError("[ERROR] 'dropout_rate' must be a float, modify config!")
    if dropout_rate <= 0 or dropout_rate >= 1:
        raise ValueError("[ERROR] 'dropout_rate' must be in the range of ]0.0, 1.0[, modify config!")
    return dropout_rate


def get_linear_units(c3d_config):
    linear_units = c3d_config["linear_units"]
    if linear_units <= 0:
        raise ValueError("[ERROR] 'linear_units' must be greater than 0, modify config!")
    return linear_units