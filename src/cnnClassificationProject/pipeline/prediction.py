import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class PredictionPipeline:
    def __init__(self,filename):
        self.filename= filename

    def predict(self):
        #load model
        # model = load_model(os.path.join("artifacts","training","model.h5"))
        model_path = os.path.abspath(os.path.join("model", "model.h5"))
        model = load_model(model_path)

        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size = (224,224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            
            prediction = "Normal"
            return [{"image" : prediction}]
        else:
            prediction = "Adenocarcinoma Cancer"
            return [{"image" : prediction}]