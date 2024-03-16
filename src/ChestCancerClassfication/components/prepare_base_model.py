import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.ChestCancerClassfication.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize the PrepareBaseModel class with the provided configuration.

        Args:
            config (PrepareBaseModelConfig): An instance of PrepareBaseModelConfig containing configuration parameters.
        """

        self.config = config

    def get_base_model(self):
        """
        Initialize the base model according to the configuration parameters and save it.
        """
        self.model = tf.keras.applications.InceptionV3(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )

        self.save_model(path = self.config.base_model_path, model = self.model)
        
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare and compile the full model based on the provided base model and configuration parameters.

        Args:
            model (tf.keras.Model): The base model.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all layers of the base model.
            freeze_till (int or None): Number of layers to freeze from the end of the base model.
                If None, no layers are frozen.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: The compiled full model.
        """

        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units = classes,
            activation = "softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = prediction
        )

        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Update the base model by preparing a new full model based on the current base model and configuration parameters,
        and save the updated model.
        """
        self.full_model = self._prepare_full_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate
        )

        self.save_model(path = self.config.updated_base_model_path, model = self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the provided model to the specified path.

        Args:
            path (Path): Path where the model will be saved.
            model (tf.keras.Model): The model to be saved.
        """
        model.save(path)
