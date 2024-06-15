import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from ChestCancerClassfication.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        # self.model_path = model_path

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        # self.model = tf.keras.Model.load_weights(self.model_path)

    def train_valid_generator(self):

        ###################################### Orginial code #########################################
        # datagenerator_kwargs = dict(
        #     rescale = 1./255
        # )

        # dataflow_kwargs = dict(
        #     target_size = self.config.params_image_size[:-1],
        #     batch_size = self.config.params_batch_size,
        #     interpolation = "bilinear"
        # )

        # if self.config.params_is_augmented:
        #     train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        #         rotation_range=0.4,
        #         horizontal_flip=True,
        #         width_shift_range=0.2,
        #         height_shift_range=0.2,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        #         fill_mode = 'nearest',
        #         **datagenerator_kwargs
        #     )
        # else:
        #     train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        #         **datagenerator_kwargs
        #     )

        # self.train_generator = train_datagenerator.flow_from_directory(
        #     directory = self.config.training_data,
        #     shuffle=True,
        #     **dataflow_kwargs
        # ) 

        # # No augmentation for validation set
        # valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        #     **datagenerator_kwargs
        # )

        # self.valid_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.valid_data,
        #     shuffle=False,
        #     **dataflow_kwargs
        # )

        ###################################### Orginial code #########################################
        datagenerator_kwargs_train = dict(
            rescale=1./255,
            rotation_range=0.4,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )

        datagenerator_kwargs_valid = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Training data generator with augmentation
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs_train
        )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=True,
            **dataflow_kwargs
        )

        # Validation data generator without augmentation
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs_valid
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.valid_data,
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save_weights(path)
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
    )