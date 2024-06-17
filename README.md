# Cancer Classification Using MLflow and DVC
The Chest Cancer Classification project is a comprehensive machine-learning application designed to diagnose different types of chest cancer from medical images. Utilizing state-of-the-art deep learning techniques and a well-structured data pipeline, this project aims to provide accurate and reliable cancer classifications, which are crucial for timely and effective treatment. 

MLflow is integrated for tracking experiments and managing model lifecycle, while DVC (Data Version Control) is used to version control data and machine learning models, ensuring reproducibility and collaboration. The backend is implemented using Flask, a lightweight web framework, to handle image uploads, preprocessing, and interfacing with the prediction pipeline. The application is containerized using Docker, allowing for consistent deployment across different environments. The deployment process is automated using a CI/CD pipeline with GitHub Actions and AWS, ensuring seamless and reliable updates to the application. 

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

- __Frontend__
    - __Description__: The frontend interface allows users to upload medical images and view the classification results. Users can interact with the application through a web interface where they can easily upload images, trigger the prediction process, and see the output.

    - __Tools__: HTML, CSS, JavaScript, Bootstrap.

- __Backend__
    - __Description__: The backend is implemented using Flask, a lightweight web framework for Python. It handles the image upload, preprocessing, and interfacing with the prediction pipeline to return the classification results to the frontend.

    - __Tools__: Flask, Python.

- __Deployment__
    - __Description__: The deployment process involves setting up a CI/CD pipeline using GitHub Actions and AWS services. This ensures that the application is built, tested, and deployed automatically whenever changes are made to the codebase. The deployment pipeline handles the following steps:

        - __Build Docker Image__: The source code is packaged into a Docker image.
        - __Push to ECR__: The Docker image is pushed to Amazon Elastic Container Registry (ECR).
        - __Launch EC2 Instance__: An Amazon EC2 instance is launched to host the application.
        - __Pull Docker Image__: The EC2 instance pulls the Docker image from ECR.
        - __Run Docker Image__: The Docker image is run on the EC2 instance, making the application accessible.

    - __Tools__: GitHub Actions, AWS (EC2, ECR), Docker.

## Overview of the Workflow 

1. [Setting up Initial Directory structure and file](#1-setting-up-initial-directory-structure-and-file)
2. [Requirements Installation](#2-requirements-installation)
3. [Logging and Exception](#3-logging-and-exception)
4. [Creating frequently used functionalities in Utils](#4-creating-frequently-used-functionalities-in-utils)
5. [Pipeline](#5-pipeline)<br>
    5.1. [Data Ingestion](#51-data-ingestion)<br>
    5.2. [Preparing the model](#52-prepare-base-model)<br>
    5.3. [Training the model](#53--training-the-model)<br>
    5.4. [Model Evaluation](#54--model-evaluation)<br>
    5.5. [Adding Data Versioning Control (DVC) to track pipeline](#55--adding-data-versioning-control-dvc-to-track-pipeline)<br>
6. [Prediction](#6-prediction)
7. [Flask Application for the Prediction Pipeline](#7-flask-application-for-the-prediction-pipeline)
8. [Dockerization and AWS CI/CD Deployment](#8-dockerization-and-aws-cicd-deployment)

## Detailed Workflow

### 1. Setting up Initial Directory structure and file
```template.py``` is designed to set up the initial directory structure and files for any project. It ensures that all necessary directories and files are created, making it easier to maintain a consistent project structure.

### 2. Requirements Installation

In this step, all the necessary dependencies required for the ```ChestCancerClassification``` project were installed in a virtual or new environment. This ensured that the project environment was properly set up with all the needed libraries and packages. The dependencies that were used for installation were listed in the requirements.txt file.

The ```-e .``` line in the ```requirements.txt``` file was a special entry that instructed pip to install the project itself as a Python package in editable mode. This meant that any changes made to the project code would immediately take effect without needing to reinstall the package. This was particularly useful during development, as it allowed for a more efficient workflow.

__Note:__ The ```__init__.py``` file is needed in each folder to turn the directory into a Python package.

The ```setup.py``` script is a crucial part of the packaging and distribution process for Python projects, ensuring that the package can be easily installed and used by others.

### 3. Logging and Exception

#### Logging

The custom logging function has been initialized in the ```src/ChestCancerClassification``` path. This function is designed to capture and record log messages, which are saved in the ```running_logs.log``` file. This logging mechanism is crucial for future debugging and monitoring of the application's behaviour.

#### Exception Handling with Python Box

The ```python-box``` library provides a convenient way to handle exceptions using the ```Box``` class. This class allows for nested dictionary-like objects that support dot notation and can be used to structure and access data more intuitively. When it comes to exception handling, ```python-box``` can help manage configuration data and other structured information in a way that enhances readability and maintainability.

### 4. Creating frequently used functionalities in Utils

- __```read_yaml```__:
The ```read_yaml``` function reads a YAML file from a specified path and returns its contents as a ```ConfigBox``` object. If the YAML file is empty, it raises a ```ValueError```. This function is essential for loading configuration settings and parameters in a structured format, making them easily accessible throughout the project.

- __```create_directories```__:
The ```create_directories``` function creates multiple directories specified in a list. It uses the ```os.makedirs``` method to ensure that all directories are created, even if they already exist. When the ```verbose``` parameter is set to ```True```, it logs the creation of each directory, providing feedback during the setup process.

- __```save_json```__:
The ```save_json``` function saves a dictionary as a JSON file at a given path. It uses Python’s built-in ```json``` module to serialize the dictionary and write it to the specified file, formatting the output for readability. This function is useful for storing configuration data, results, or any structured data in a standard format.

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
In this stage, the VGG16 model is downloaded and prepared for transfer learning. The pre-trained model is obtained, and its convolutional layers are left unchanged (essentially frozen). The fully connected layers are replaced with custom layers to accommodate the specific classes to be predicted. This updated model is stored before being trained. 

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

Note: The model could be large. Make sure you use [Git Large File Storage](https://git-lfs.com/) to push the model.

#### 5.4.  Model Evaluation:

The ```Evaluation``` class is designed to handle the evaluation process of a trained TensorFlow model for the ```ChestCancerClassification``` project. This class includes methods to set up a validation data generator, load a trained model, evaluate the model, save the evaluation scores, and log the evaluation results into MLflow.

- __Connecting with [Dagshub](https://dagshub.com/docs/index.html)__:<br>
DagsHub is a platform for AI and ML developers that lets you manage and collaborate on your data, models, and experiments, alongside your code. Dagshub is integrated with [MLflow](https://mlflow.org/docs/latest/index.html) to keep track of experimentation process, log parameters and metrics. 

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

    - The preprocessed image is passed to the model’s prediction function, which returns the prediction probabilities for each class.
    - The class with the highest probability is determined using np.argmax.

- __Interpreting Results__:

    - The predicted class index is mapped to the corresponding cancer type:
    0 → Adenocarcinoma Cancer<br>
    1 → Large Cell Carcinoma Cancer<br>
    2 → Normal<br>
    Other → Squamous Cell Carcinoma Cancer<br>
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

### 8. Dockerization and AWS CI/CD Deployment
The ```main.yaml``` file is set up to perform CI/CD using GitHub Actions with AWS. Here's a breakdown of the steps mentioned:

- __Continuous Integration (CI) and Continuous Delivery (CD)__:

    - __CI/CD in GitHub Actions__: The file defines a CI/CD pipeline using GitHub Actions. GitHub Actions is used to automate the building, testing, and deployment of code.

- __Connecting with AWS:__

    - The script connects with AWS, and these AWS credentials are configured to allow GitHub Actions to interact with AWS services.

- __Installing Necessary Libraries:__

    - The script installs the necessary libraries required for building and deploying the application. This includes tools like the AWS CLI, Docker, or other dependencies.

- __Logging in to Amazon ECR (Elastic Container Registry):__

    - __ECR__: Amazon Elastic Container Registry (ECR) is a managed Docker container registry that makes it easy to store, manage, and deploy Docker container images.
    - The script logs into ECR using AWS credentials to gain access to the container registry.

- __Building the Docker Image__:

    -The script builds a Docker image for the application. This involves using a Dockerfile to package the application and its dependencies into a container image.

- __Pushing the Docker Image to ECR:__

    - After building the Docker image, the script pushes the image to Amazon ECR. This makes the image available for deployment.

- __Continuous Deployment (CD):__

    - __Logging into AWS__: In the deployment step, the workflow logs into AWS again to perform deployment actions.
    - __Pulling the Docker Image from ECR__: The workflow pulls the Docker image from Amazon ECR to ensure it has the latest version of the container.
    - __Running the Docker Image on EC2__: The workflow runs the Docker image on an EC2 instance. EC2 (Elastic Compute Cloud) is a service that provides resizable compute capacity in the cloud, essentially virtual servers.

## Apendix

### Brief on AWS-CICD-Deployment-with-GitHub-Actions Steps
This guide outlines the steps to set up a CI/CD pipeline for deploying a Dockerized application to AWS using GitHub Actions. The process involves building and pushing a Docker image from the source code, pushing it to Amazon ECR, launching an EC2 instance, pulling the Docker image from ECR, and running the Docker image on the EC2 instance.

Steps to set up AWS CI/CD deployment with GitHub Action:

- __Login to AWS console__: Gain access to the AWS Management Console to perform necessary configurations and create resources.
- __Create IAM user for deployment__: Create a dedicated IAM user with specific permissions for CI/CD operations.
    - __EC2 Access__: Allows managing EC2 instances, which are virtual machines.
    - __ECR Access__: Allows managing Amazon Elastic Container Registry, used to store Docker images.
- __Required IAM Policies__:
    - __AmazonEC2ContainerRegistryFullAccess__: Grants full access to ECR.
    - __AmazonEC2FullAccess__: Grants full access to EC2 resources.
- __Create ECR Repository__: Create ECR Repository.
- __Create EC2 Machine (Ubuntu)__: Launch an EC2 instance to host and run the Dockerized application.
- __Install Docker on EC2 Machine__:
    - __Update and upgrade the EC2 instance__:
    ```bash
    sudo apt-get update -y
    sudo apt-get upgrade
    ```
    - __Install Docker__:
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    newgrp docker
    ```
- __Configure EC2 as Self-Hosted Runner__: Use the EC2 instance as a self-hosted runner for GitHub Actions.
    - Go to ```Settings > Actions > Runners``` in your GitHub repository.
    - Add a new self-hosted runner, choose the OS (Linux), and follow the provided commands to set it up on the EC2 instance.
- __Setup GitHub Secrets__: Store sensitive information required for deployment as secrets in GitHub.
    - __AWS_ACCESS_KEY_ID__: Your AWS Access Key ID.
    - __AWS_SECRET_ACCESS_KEY__: Your AWS Secret Access Key.
    - __AWS_REGION__: Region where your resources are located, e.g., us-east-1.
    - __AWS_ECR_LOGIN_URI__: URI for logging into ECR, e.g., 5663734*****.dkr.ecr.us-east-1.amazonaws.com.
    - __ECR_REPOSITORY_NAME__: Name of your ECR repository, e.g., simple-app.

## License

This project is licensed under the MIT License.

