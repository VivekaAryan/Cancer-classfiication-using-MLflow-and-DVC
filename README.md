# Cancer Classfiication Using MLflow and DVC

## Workflow 

1. Setting up Initial Directory structure and file
2. Requirements Installation
3. Logging and Exception
4. Creating frequently used functionalities in Utils
5. Update the configuration manager in src config
6. Udpate the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

## Workflow

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