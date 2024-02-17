import tensorflow as tf
import json
from keras import Model, Input
from keras.layers import Conv3D, ReLU, MaxPool3D, Flatten, Dropout, Dense, Softmax
from keras.activations import sigmoid
from pathlib import Path
        

configs_dir = Path.cwd().joinpath("configs")

     
def C3D():
    c3d_config_path = configs_dir.joinpath("C3D.json")
    
    if not c3d_config_path.exists():
        raise ValueError("[ERROR] configs/C3D.json does not exist!")
    
    with open(c3d_config_path, "r") as f:
        config = json.load(f)
        
    input_layer = Input(shape=(config["capacity"], config["image-width"], config["image-height"], config["color-channels"]))
    
    conv_1 = Conv3D(filters=config["conv-1-filters"], kernel_size=config["conv-kernel-size"], strides=config["conv-stride"], padding=config["conv-padding"])(input_layer)
    relu_1 = ReLU()(conv_1)
    pool_1 = MaxPool3D(pool_size=config["pool-1-size"])(relu_1)
    
    conv_2_a = Conv3D(filters=config["conv-2-filters"], kernel_size=config["conv-kernel-size"], strides=config["conv-stride"], padding=config["conv-padding"])(pool_1)
    relu_2_a = ReLU()(conv_2_a)
    conv_2_b = Conv3D(filters=config["conv-2-filters"], kernel_size=config["conv-kernel-size"], strides=config["conv-stride"], padding=config["conv-padding"])(relu_2_a)
    relu_2_b = ReLU()(conv_2_b)
    pool_2 = MaxPool3D(pool_size=config["pool-rest-size"])(relu_2_b)
    
    conv_3_a = Conv3D(filters=config["conv-3-filters"], kernel_size=config["conv-kernel-size"], strides=config["conv-stride"], padding=config["conv-padding"])(pool_2)
    relu_3_a = ReLU()(conv_3_a)
    conv_3_b = Conv3D(filters=config["conv-3-filters"], kernel_size=config["conv-kernel-size"], strides=config["conv-stride"], padding=config["conv-padding"])(relu_3_a)
    relu_3_b = ReLU()(conv_3_b)
    pool_3 = MaxPool3D(pool_size=config["pool-rest-size"])(relu_3_b)
    
    conv_4_a = Conv3D(filters=config["conv-4-filters"], kernel_size=config["conv-kernel-size"], strides=config["conv-stride"], padding=config["conv-padding"])(pool_3)
    relu_4_a = ReLU()(conv_4_a)
    conv_4_b = Conv3D(filters=config["conv-4-filters"], kernel_size=config["conv-kernel-size"], strides=config["conv-stride"], padding=config["conv-padding"])(relu_4_a)
    relu_4_b = ReLU()(conv_4_b)
    pool_4 = MaxPool3D(pool_size=config["pool-rest-size"])(relu_4_b)
    
    flatten = Flatten()(pool_4)
    
    dropout_1 = Dropout(config["dropout-pct"])(flatten)
    fc_6 = Dense(config["linear-units"], name="fc_6")(dropout_1)
    relu_6 = ReLU(name="relu_fc_6")(fc_6)
    
    dropout_2 = Dropout(config["dropout-pct"])(relu_6)
    fc_7 = Dense(config["linear-units"], name="fc_7")(dropout_2)
    relu_7 = ReLU(name="relu_fc_7")(fc_7)
    
    if config["num-classes"] == 2:
        fc_8 = Dense(1)(relu_7)
        output_layer = sigmoid(fc_8)
    else:
        fc_8 = Dense(config["num-classes"])(relu_7)
        output_layer = Softmax()(fc_8)
        
    return Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":
    model = C3D()
    
    for layer in model.layers:
        print(layer)