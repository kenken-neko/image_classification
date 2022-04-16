import tensorflow as tf
import random


class TFImagePreprocessing:
    def __init__(
        self,
        hight_size,
        width_size,
        channel_size,
        num_classes,
    ):
        self._image_size = [hight_size, width_size]
        self._channel_size = channel_size
        self._num_classes = num_classes

    def rand_augment(self, image, seed=None):
        image = tf.image.random_flip_left_right(image, seed)
        image = tf.image.random_brightness(image, 0.25, seed)
        return image

    def base_preprocess(self, dataset):
        image = dataset["image"]
        image = tf.cast(image, tf.float32)
        image /= 255.0
        label = tf.one_hot(dataset["label"], self._num_classes)
        return image, label

    def vgg_preprocess(self, dataset):
        # VGG input image size and channel
        input_size, input_channel = [224, 224], 3
        image, label = self.base_preprocess(dataset)
        if self._image_size != input_size:
            image = tf.image.resize(image, input_size)
        if self._channel_size != input_channel:
            image = tf.image.grayscale_to_rgb(image)
        return image, label

    def xception_preprocess(self, dataset):
        # Xception input image size and channel
        input_size, input_channel = [299, 299], 3
        image, label = self.base_preprocess(dataset)
        if self._image_size != input_size:
            image = tf.image.resize(image, input_size)
        if self._channel_size != input_channel:
            image = tf.image.grayscale_to_rgb(image)
        return image, label
