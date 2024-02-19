from keras import Model, Input
from keras.layers import ReLU, Dense, Dropout, Softmax


def classifier(feat_dim = 1024, num_units = 1024, num_classes = 6, dropout_rate = 0.5):
    input_layer = Input(shape=(feat_dim,))
    
    fc_1 = Dense(num_units)(input_layer)
    relu_1 = ReLU()(fc_1)
    drop_1 = Dropout(dropout_rate)(relu_1)
    
    fc_2 = Dense(num_units)(drop_1)
    relu_2 = ReLU()(fc_2)
    drop_2 = Dropout(dropout_rate)(relu_2)
    
    output_layer = Dense(num_classes)(drop_2)
    output_layer = Softmax()(output_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
    


if __name__ == "__main__":
    model = classifier()
    model.summary()