from keras.optimizers import SGD, Adam


def get_modality(c3d_config):
    if c3d_config["color-channels"] == 2:
        return "flow"
    elif c3d_config["color-channels"] == 3:
        return "rgb"
    else:
        raise ValueError("[ERROR] 'color-channels' can only be 2 or 3, modify the config file!")

   
def get_task(c3d_config):
    if c3d_config["task"] not in ["binary", "multi"]:
        raise ValueError("[ERROR] 'task' can only be 'binary' or 'multi', modify the config file!")
    return c3d_config["task"]


def get_num_classes(c3d_config):
    if c3d_config["num-classes"] <= 0:
        raise ValueError("[ERROR] 'num-classes' must be greater than 0, modify the config file!")
    return c3d_config["num-classes"]


def get_input_shape(c3d_config):
    l = c3d_config["capacity"]
    w = c3d_config["image-width"]
    h = c3d_config["image-height"]
    c = c3d_config["color-channels"]
    
    if l < 8 or l > 16:
        raise ValueError("[ERROR] 'capacity' must be between 8 and 16, modify the config file!")
    if (w < 96 or w > 224) and (h < 96 or h > 224) and w != h:
        raise ValueError(
            "[ERROR] use spatial dimensions with shape (N, N), where\n"
            "N is >= 96 and <= 224, modify the config file!"
        )
    if c not in [2, 3]:
        raise ValueError("[ERROR] 'color-channels' must be 2 or 3, modify the config file!")
    
    return (l, w, h, c)


def get_output_shape(c3d_config):
    if get_task(c3d_config) == "binary":
        return (1,)
    else:
        return (get_num_classes(c3d_config),)   


def get_epochs(c3d_config):
    if c3d_config["epochs"] < 0:
        raise ValueError("[ERROR] 'epochs' must be greater than 0, modify the config file!")
    return c3d_config["epochs"]


def get_optimizer(c3d_config):
    optimizer = c3d_config["optimizer"].lower()
    learning_rate = c3d_config["learning-rate"]
    
    if optimizer not in ["adam", "sgd"]:
        raise ValueError(f"[ERROR] Only Adam and SGD optimizers are allowed, modify the config file!")
    if learning_rate > 0.1:
        raise ValueError(f"[ERROR] 'learning-rate' is too high, modify the config file!")
    if optimizer == "adam":
        return Adam(learning_rate=learning_rate)
    if optimizer == "sgd":
        return SGD(learning_rate=learning_rate)
    
    
def get_batch_size(c3d_config):
    batch_size = c3d_config["batch-size"]
    if batch_size <= 0:
        raise ValueError("[ERROR] 'batch-size' must be greater than 0, modify the config file!")
    if not batch_size % 2 == 0:
        raise ValueError("[ERROR] 'batch-size' must be divisable by 2, modify the config file!")
    return batch_size


def get_dropout(c3d_config):
    dropout_pct = c3d_config["dropout-pct"]
    if not isinstance(dropout_pct, float):
        raise ValueError("[ERROR] 'dropout-pct' must be a float, modify the config file!")
    if dropout_pct <= 0 or dropout_pct >= 1:
        raise ValueError("[ERROR] keep 'dropout-pct' between ]0,1[!")
    return dropout_pct


def get_linear_units(c3d_config):
    linear_units = c3d_config["linear-units"]
    if linear_units <= 0:
        raise ValueError("[ERROR] 'linear-units' must be greater than 0, modify the config file!")
    return linear_units