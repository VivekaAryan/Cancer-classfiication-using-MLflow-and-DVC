import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ChestCancerClassfication import logger
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))
        # model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (299,299))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image / 255.0

        prediction = model.predict(test_image)
        result = np.argmax(prediction, axis=1)
        logger.info(f"Prediction result index: {result}")
        logger.info(f"Raw prediction: {np.round(prediction,2)}")

        if result[0] == 0:
            prediction = 'Adenocarcinoma Cancer'
        elif result[0] == 1:
            prediction = 'Large Cell Carcinoma Cancer'
        elif result[0] == 2:
            prediction = 'Normal'
        else:
            prediction = 'Squamous Cell Carcinoma Cancer'
        return [{"image": prediction}]
