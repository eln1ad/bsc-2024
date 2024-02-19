from keras.optimizers import SGD, Adam


def get_modality(config):
    if config["modality"] not in ["rgb", "flow"]:
        raise ValueError("[ERROR] 'modality' can only be 'rgb' or 'flow', modify config!")
    return config["modality"]


def get_epochs(config):
    if config["epochs"] < 0:
        raise ValueError("[ERROR] 'epochs' must be greater than 0, modify the config file!")
    return config["epochs"]


def get_input_shape(config):
    if config["feature_dim"] <= 256:
        raise ValueError("[ERROR] 'feature_dim' must be at least 256, modify config!")
    return (config["feature_dim"],)


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


def get_linear_units(config):
    linear_units = config["linear_units"]
    if linear_units <= 0:
        raise ValueError("[ERROR] 'linear-units' must be greater than 0, modify config!")
    return linear_units