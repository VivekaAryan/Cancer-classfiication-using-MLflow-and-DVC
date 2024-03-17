from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from ChestCancerClassfication.utils.common import decodeImage
from ChestCancerClassfication.pipeline.step_05_prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"                            # Saving the input image
        self.classifier = PredictionPipeline(self.filename)         # Running the prediction pipeline on the input image

# Landing page
@app.route("/", methods = ['GET'])
@cross_origin()
def home():
    return render_template('index.html')         

# Training Route doen by calling main.py
@app.route("/train", methods = ['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")                   
    # os.system("dvc repro")                   
    return "Training done successfully!"

# Prediction Route
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host = '0.0.0.0', port = 8080)      # AWS
