from keras import Model, Input
from keras.layers import ReLU, Dense, Dropout
from keras.activations import sigmoid


def detector(feat_dim=1024, num_units=1024, dropout_rate=0.5):
    input_layer = Input(shape=(feat_dim,))
    
    fc_1 = Dense(num_units)(input_layer)
    relu_1 = ReLU()(fc_1)
    drop_1 = Dropout(dropout_rate)(relu_1)
    
    fc_2 = Dense(num_units)(drop_1)
    relu_2 = ReLU()(fc_2)
    drop_2 = Dropout(dropout_rate)(relu_2)
    
    out_confidence = Dense(1)(drop_2)
    out_confidence = sigmoid(out_confidence)
    
    # locations won't have an activation function
    out_center = Dense(1)(drop_2)
    out_length = Dense(1)(drop_2)
    
    model = Model(inputs=input_layer, outputs=[out_confidence, out_center, out_length])
    return model


if __name__ == "__main__":
    model = detector()
    model.summary()