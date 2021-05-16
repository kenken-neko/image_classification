import tensorflow as tf


class ImagePreprocess:

    def __init__(
        self, 
        channel_size,
        num_classes,
    ):
        self._channel_size = channel_size
        self._num_classes = num_classes

    def base_preprocess(self, dataset):
        images = dataset["image"]
        images = tf.cast(images, tf.float32)
        images /= 255.0
        labels = tf.one_hot(dataset["label"], self._num_classes)
        return images, labels

    def vgg_preprocess(self, dataset):
        images, labels = self.base_preprocess(dataset)
        images = tf.image.resize(
            images, [224, 224]
        )
        if self._channel_size == 1:
            images = tf.image.grayscale_to_rgb(images)
        return images, labels

    def xception_preprocess(self, dataset):
        images, labels = self.base_preprocess(dataset)
        images = tf.image.resize(
            images, [299, 299]
        )
        if self._channel_size == 1:
            images = tf.image.grayscale_to_rgb(images)
        return images, labels
