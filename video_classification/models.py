from keras import Model, Input
from keras.layers import Conv3D, ReLU, MaxPool3D, Flatten, Dropout, Dense, Softmax
from keras.activations import sigmoid

     
def C3D(input_shape = None, num_classes = 2, linear_units = 1024, dropout_pct = 0.5,):
    input_layer = Input(shape=input_shape)
    
    conv_1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(input_layer)
    relu_1 = ReLU()(conv_1)
    pool_1 = MaxPool3D(pool_size=(1, 2, 2))(relu_1)
    
    conv_2_a = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(pool_1)
    relu_2_a = ReLU()(conv_2_a)
    conv_2_b = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(relu_2_a)
    relu_2_b = ReLU()(conv_2_b)
    pool_2 = MaxPool3D(pool_size=(2, 2, 2))(relu_2_b)
    
    conv_3_a = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(pool_2)
    relu_3_a = ReLU()(conv_3_a)
    conv_3_b = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(relu_3_a)
    relu_3_b = ReLU()(conv_3_b)
    pool_3 = MaxPool3D(pool_size=(2, 2, 2))(relu_3_b)
    
    conv_4_a = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(pool_3)
    relu_4_a = ReLU()(conv_4_a)
    conv_4_b = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(relu_4_a)
    relu_4_b = ReLU()(conv_4_b)
    pool_4 = MaxPool3D(pool_size=(2, 2, 2))(relu_4_b)
    
    flatten = Flatten()(pool_4)
    
    dropout_1 = Dropout(dropout_pct)(flatten)
    fc_6 = Dense(linear_units, name="fc_6")(dropout_1)
    relu_6 = ReLU(name="relu_fc_6")(fc_6)
    
    dropout_2 = Dropout(dropout_pct)(relu_6)
    fc_7 = Dense(linear_units, name="fc_7")(dropout_2)
    relu_7 = ReLU(name="relu_fc_7")(fc_7)
    
    if num_classes == 2:
        fc_8 = Dense(1)(relu_7)
        output_layer = sigmoid(fc_8)
    else:
        fc_8 = Dense(num_classes)(relu_7)
        output_layer = Softmax()(fc_8)
        
    return Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":
    model = C3D(input_shape=(8, 112, 112, 3))
    
    for layer in model.layers:
        print(layer)