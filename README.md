# Cancer Classfiication Using MLflow and DVC
The Chest Cancer Classification project is a comprehensive machine learning application designed to diagnose different types of chest cancer from medical images. Utilizing state-of-the-art deep learning techniques and a well-structured data pipeline, this project aims to provide accurate and reliable cancer classifications, which are crucial for timely and effective treatment.

## Key Components
- __Data Ingestion__

    - __Description__: This stage involves downloading and preparing the raw dataset for further processing. The dataset is fetched from a specified URL, stored locally, and extracted for use in subsequent stages.
    - __Tools__: DVC, YAML for configuration.

- __Data Preprocessing__

    - __Description__: The raw data undergoes cleaning and transformation to ensure it is in a suitable format for model training. This includes resizing images, normalizing pixel values, and potentially augmenting the data to enhance the model's robustness.
    - __Tools__: TensorFlow, Python scripts.
    
- __Model Training__

    - __Description__: A deep learning model (VGG16) is utilized for transfer learning. The pre-trained model’s convolutional layers are frozen, and custom fully connected layers are added to classify the specific types of chest cancer. The model is trained on the preprocessed dataset.
    - __Tools__: TensorFlow, Keras.

- __Model Evaluation__

    - __Description__: The trained model is evaluated on a validation dataset to measure its performance. Metrics such as accuracy and loss are computed and logged.
    - __Tools__: TensorFlow, MLflow for experiment tracking.

- __Prediction Pipeline__

    - __Description__: This pipeline handles the classification of new medical images. It loads the trained model, processes input images, and outputs predictions indicating the type of chest cancer.
    - __Tools__: TensorFlow, Numpy.
    
## Overview of the Workflow 

1. [Setting up Initial Directory structure and file](#1-setting-up-initial-directory-structure-and-file)
2. [Requirements Installation](#2-requirements-installation)
3. [Logging and Exception](#3-logging-and-exception)
4. [Creating frequently used functionalities in Utils](#4-creating-frequently-used-functionalities-in-utils)
5. [Pipeline](#5-pipeline)
    5.1. [Data Ingestion](#51-data-ingestion)
    5.2. [Preparing the model](#52-prepare-base-model)
    5.3. [Training the model](#53--training-the-model)
    5.4. [Model Evaluation](#54--model-evaluation)
    5.5. [Adding Data Versioning Control (DVC) to track pipeline](#55--adding-data-versioning-control-dvc-to-track-pipeline)
6. [Prediction](#6-prediction)

## Detailed Workflow

### 1. Setting up Initial Directory structure and file
```template.py``` is designed to set up the initial directory structure and files for any project. It ensures that all necessary directories and files are created, making it easier to maintain a consistent project structure.

### 2. Requirements Installation

In this step, all the necessary dependencies required for the ```ChestCancerClassification``` project were installed in a virtual or new environment. This ensured that the project environment was properly set up with all the needed libraries and packages. The dependencies were listed in the requirements.txt file, which was used for installation.

The ```-e .``` line in the ```requirements.txt``` file was a special entry that instructed pip to install the project itself as a Python package in editable mode. This meant that any changes made to the project code would immediately take effect without needing to reinstall the package. This was particularly useful during development, as it allowed for a more efficient workflow.

__Note:__ The ```__init__.py``` file is needed in each folder to turn the directory into a Python package.

The ```setup.py``` script is a crucial part of the packaging and distribution process for Python projects, ensuring that the package can be easily installed and used by others.

### 3. Logging and Exception

#### Logging

The custom logging function has been initialized in the ```src/ChestCancerClassification``` path. This function is designed to capture and record log messages, which are saved in the ```running_logs.log``` file. This logging mechanism is crucial for future debugging and monitoring the application's behavior.

#### Exception Handling with Python Box

The ```python-box``` library provides a convenient way to handle exceptions using the ```Box``` class. This class allows for nested dictionary-like objects that support dot notation and can be used to structure and access data more intuitively. When it comes to exception handling, ```python-box``` can help manage configuration data and other structured information in a way that enhances readability and maintainability.

### 4. Creating frequently used functionalities in Utils

- __```read_yaml```__:
The ```read_yaml``` function reads a YAML file from a specified path and returns its contents as a ```ConfigBox``` object. If the YAML file is empty, it raises a ```ValueError```. This function is essential for loading configuration settings and parameters in a structured format, making them easily accessible throughout the project.

- __```create_directories```__:
The ```create_directories``` function creates multiple directories specified in a list. It uses the ```os.makedirs``` method to ensure that all directories are created, even if they already exist. When the ```verbose``` parameter is set to ```True```, it logs the creation of each directory, providing feedback during the setup process.

- __```save_json```__:
The ```save_json``` function saves a dictionary as a JSON file at a given path. It uses Python’s built-in ```json``` module to serialize the dictionary and writes it to the specified file, formatting the output for readability. This function is useful for storing configuration data, results, or any structured data in a standard format.

- __```save_bin```__:
The ```save_bin``` function saves data as a binary file using the ```joblib``` library. This function is designed to handle the serialization of large datasets or models, ensuring they can be efficiently stored and later retrieved without data loss. It logs the path where the binary file is saved for easy tracking.

- __```load_bin```__:
The ```load_bin``` function loads data from a binary file, also using the ```joblib``` library. This function is essential for deserializing large datasets or models that were previously saved, enabling them to be used in subsequent processing steps. It logs the loading process to confirm successful retrieval.

- __```get_size```__:
The ```get_size``` function returns the size of a file in kilobytes (KB). It uses the ```os.path.getsize``` method to get the file size in bytes and then converts it to kilobytes. This function provides a quick way to check the size of files, which can be useful for monitoring storage usage and managing resources.

- __```decodeImage```__:
The ```decodeImage``` function decodes a base64-encoded string and saves it as an image file. It converts the base64 string back into binary image data and writes it to a file specified by the ```fileName``` parameter. This function is useful for handling image data transmitted in base64 format, such as images received from web APIs.

- __```encodeImageIntoBase64```__:
The ```encodeImageIntoBase64``` function encodes an image file into a base64 string. It reads the binary data of an image file and converts it into a base64-encoded string. This function is useful for preparing image data to be transmitted or stored in text-based formats, such as embedding images in JSON or HTML documents.

Note: 
- The ```ensure_annotations``` decorator from the ```ensure``` library is used to enforce type annotations in Python functions. It ensures that the arguments passed to a function match the specified type annotations, and it can also validate the return type of the function. This helps in catching type-related errors early, making the code more robust and easier to understand.

- The ```ConfigBox``` class from the ```python-box``` library extends the standard dictionary to provide additional features, such as dot notation access. This makes it easier to work with nested dictionary structures, as you can access elements using attribute-style access instead of key-based access.



### 5. Pipeline

For each stage in the pipeline, you:
- Add the necessary paths to ```config\congfig.yaml```.
- Update param.yaml for "Stage 2-Preparing the base model".
- Create an entity in ```src\ChestCancerClassfication\entity```
- Update the configmanager in ```src\ChestCancerClassfication\config```
- Update component in ```src\ChestCancerClassfication\components```
- Update pipeline in ```src\ChestCancerClassfication\pipeline```
- Update main.py
- Update teh DVC.yaml 

#### 5.1. Data Ingestion:

This stage involves downloading the dataset from a specified URL, storing it locally, and extracting the contents for further use in the project. 

Using a YAML configuration makes the process easily configurable and maintainable, allowing for flexibility in specifying different datasets or storage locations. In this approach, configuration details are neatly separated from the code, resulting in cleaner and more adaptable code. This method offers significant benefits in terms of maintainability, scalability, and collaboration.

This YAML code is wirtten in ```config\config.yaml```. The data is downloaded and extracted and stored in the artifacts folder. 

Note: The testing is all done in the research folder. Also, make sure the data is added to ```.gitignore```.

#### 5.2. Prepare Base Model:
In this stage, the VGG16 model is downloaded and prepared for transfer learning. The pre-trained model is obtained, and its convolutional layers are left unchanged (essentially frozen). The fully connected layers are replaced with custom layers to accommodate the specific classes to be predicted. This updated model is stored befoer being trained. 

Note: Details of each model can be found in [Keras Documentation](https://keras.io/api/applications/).

#### 5.3.  Training the model:
This ```Training``` class encapsulates the entire training process, including loading the pre-trained model, setting up data generators with optional augmentation, training the model, and saving the trained model. The class is designed to be flexible and configurable, allowing for easy adjustments through the ```TrainingConfig``` object.

- __Initialization__: Sets up the ```Training``` object with necessary configurations.
    - __Action__: Receives a ```TrainingConfig``` instance containing all parameters required for training.

- __Get Base Model (get_base_model)__: Loads a pre-trained model for transfer learning.
    - __Action__: Loads the model specified in config.updated_base_model_path into self.model.

- __Train and Validation Generator (train_valid_generator)__: Sets up data generators for training and validation.
    - __Actions__:
        - __Data Augmentation__: Applies data augmentation if config.params_is_augmented is True.
        - __Data Scaling__: Rescales image pixel values to [0, 1].
        - __Training Data Generator__: Loads and augments training data from config.training_data.
        - __Validation Data Generator__: Loads validation data from config.valid_data without augmentation.

- __Save Model (save_model)__: Saves the trained model to a specified location.
    - __Action__: Saves the Keras model to the path provided.

- __Training Process (train)__: Trains the neural network using the data generators.
    - __Actions__:
        - __Steps Calculation__: Computes steps per epoch for training and validation.
        - __Model Training__: Trains the model for a set number of epochs and validation steps.
        - __Model Saving__: Saves the trained model to config.trained_model_path.

#### 5.4.  Model Evaluation:

The ```Evaluation``` class is designed to handle the evaluation process of a trained TensorFlow model for the ```ChestCancerClassification``` project. This class includes methods to set up a validation data generator, load a trained model, evaluate the model, save the evaluation scores, and log the evaluation results into MLflow.

- __Connecting with [Dagshub](https://dagshub.com/docs/index.html)__:<br>
DagsHub is a platform for AI and ML developers that lets you manage and collaborate on your data, models, experiments, alongside your code. Dagshub is integrated with [MLflow](https://mlflow.org/docs/latest/index.html) to keep track of expereimentation process, log parameters and metrics. 

    Note: The following credentials need to be exported as env variables by running in the terminal:

    ```Python
    export MLFLOW_TRACKING_URI=
    export MLFLOW_TRACKING_USERNAME= 
    export MLFLOW_TRACKING_PASSWORD=
    ```

- __Initialization__: The Evaluation class is initialized with an EvaluationConfig object containing configuration details such as paths, image size, batch size, and MLflow URI.
- __Validation Data Setup__: The _valid_generator method prepares the validation data generator with specified preprocessing steps and data flow parameters.
- __Model Loading__: The load_model method loads a pre-trained TensorFlow model from the specified path.
- __Evaluation__: The evaluation method evaluates the loaded model using the validation data generator, calculates the loss and accuracy, and saves these scores to a JSON file.
- __MLflow Logging__: The log_into_mlfow method logs the evaluation parameters and metrics into MLflow, and if applicable, registers the model.

__Note__: MLflow provides different functions for logging models from various libraries like Keras, Scikit-learn, and PyTorch. The specific method to log a model in MLflow depends on the machine learning library being used.

```Python
# For Keras
import mlflow.keras
mlflow.keras.log_model(model, "model", registered_model_name="MyKerasModel")

# For Scikit-learn
import mlflow.sklearn
mlflow.sklearn.log_model(model, "model", registered_model_name="MySklearnModel")

# For Pytorch
import mlflow.pytorch
mlflow.pytorch.log_model(model, "model", registered_model_name="MyPyTorchModel")
```

#### 5.5.  Adding Data Versioning Control (DVC) to track pipeline:

__[Data Version Control (DVC)](https://dvc.org/doc)__ is an open-source tool designed to manage machine learning projects efficiently. It offers version control for data and machine learning models, akin to how Git manages code. One of DVC's powerful features is its ability to create pipelines that define the sequence of stages in a machine learning workflow. These pipelines help ensure that each step in the workflow is reproducible and trackable.

__Benefits of Using DVC Pipelines__:<br>
    - __Reproducibility__: Ensures that every step of your machine learning process can be reproduced, from data preprocessing to model training.
    - __Versioning__: Tracks changes in data, code, and models, allowing for easy rollback and comparison.
    - __Collaboration__: Facilitates collaboration by keeping the workflow and its dependencies consistent across different environments and team members.
    - __Automation__: Automates the workflow, ensuring that any changes in the pipeline trigger the necessary stages to run.

In this project, DVC is used for pipeline tracking. This setup is done in ```dvc.yaml``` file.

How to run DVC:


```Python
# Initialize DVC
dvc init

# Run dvc repro
dvc repro
``` 

This will set up a ```.dvc```, ```dvc.lock``` and ```.dvcignore``` in your root folder.

- __```.dvc``` Directory__: Manages DVC configurations and cached data to enable version control and efficient data tracking.
- __```dvc.lock``` File__: Ensures reproducibility by locking the exact versions of data and code dependencies for each pipeline stage.
- __```.dvcignore``` File__: Excludes specified files and directories from DVC tracking, similar to .gitignore in Git.

### 6. Prediction
The ```PredictionPipeline``` class is designed to handle the prediction stage of the ```ChestCancerClassification``` project. This stage involves loading a trained neural network model, processing an input image, making predictions, and returning the prediction results.

- __Loading the Model__:
    - The model is loaded from the artifacts/training directory using. TensorFlow’s load_model function. This model is the result of the training stage and is used to make predictions on new data.

- __Image Preprocessing__:

    - The input image specified by self.filename is loaded and resized to the target size of (350, 360) pixels using image.load_img.
    - The image is then converted to a numpy array with image.img_to_array and expanded to include an additional dimension using np.expand_dims to match the input shape required by the model.

- __Making Predictions__:

    - The preprocessed image is passed to the model’s predict function, which returns the prediction probabilities for each class.
    - The class with the highest probability is determined using np.argmax.

- __Interpreting Results__:

    - The predicted class index is mapped to the corresponding cancer type:
    0 → Adenocarcinoma Cancer
    1 → Large Cell Carcinoma Cancer
    2 → Normal
    Other → Squamous Cell Carcinoma Cancer
    - The prediction is returned as a dictionary containing the predicted cancer type.

### 7. Flask Application for the Prediction Pipeline
This Flask application serves as a web interface for the ```ChestCancerClassification``` project. It provides endpoints for rendering the homepage, initiating the training process, and making predictions using a trained model. The application integrates CORS to allow cross-origin requests, enabling it to interact with other web services or clients.

- __Initialization__:

    - The environment variables ```LANG``` and ```LC_ALL``` are set to ensure the application uses the English locale.
    - The Flask application is created, and CORS is enabled to allow cross-origin requests.

- __ClientApp Class__:

    - The ```ClientApp``` class is instantiated, setting the ```filename``` for the input image and initializing the ```PredictionPipeline``` with this filename.

- __Home Route__:

    - The home route renders the ```index.html``` page, providing a user interface for interacting with the application.

- __Training Route__:

    - The training route can be triggered via a GET or POST request. It runs the training script (```main.py```) to train the model and returns a success message once training is completed.

- __Prediction Route__:

    - The prediction route accepts a POST request containing a base64-encoded image.
    - The image is decoded and saved using the decodeImage function.
    - The PredictionPipeline runs the prediction on the saved image, and the result is returned as a JSON response.