import tensorflow as tf
from pathlib import Path
import keras


# Which layer should be used in C3D? relu_fc_6
def build_feature_extractor(model_name, last_layer_name="relu_fc_6"):
    saved_model_path = Path.cwd().joinpath("saved_models").joinpath(model_name)
    
    if not saved_model_path.exists():
        raise ValueError("Model with this name does not exist!")
    
    model = keras.models.load_model(saved_model_path)
    last_layer = model.get_layer(name=last_layer_name)
    model = tf.keras.Model(inputs=model.inputs, outputs=last_layer.output)
    return model
        
        
if __name__ == "__main__":
    feature_extractor = build_feature_extractor("C3D_rgb_8_frames_40_epochs")
    feature_extractor.summary()