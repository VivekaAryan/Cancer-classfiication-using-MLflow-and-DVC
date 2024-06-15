# Cancer Classfiication Using MLflow and DVC

## Workflow

1. Update config.yaml
2. Update secrets.yaml [optional]
3. Update params.yaml
4. Update the entity
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