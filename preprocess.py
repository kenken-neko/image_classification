import tensorflow as tf


class TFImagePreprocess:

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

    def base_preprocess(self, dataset):
        images = dataset["image"]
        images = tf.cast(images, tf.float32)
        images /= 255.0
        labels = tf.one_hot(dataset["label"], self._num_classes)
        return images, labels

    def vgg_preprocess(self, dataset):
        # VGG input image size and channel
        input_size, input_channel = [224, 224], 3
        images, labels = self.base_preprocess(dataset)
        if self._image_size != input_size:
            images = tf.image.resize(
                images, input_size
            )
        if self._channel_size != input_channel:
            images = tf.image.grayscale_to_rgb(images)
        return images, labels

    def xception_preprocess(self, dataset):
        # Xception input image size and channel
        input_size, input_channel = [299, 299], 3
        images, labels = self.base_preprocess(dataset)
        if self._image_size != input_size:
            images = tf.image.resize(
                images, input_size
            )
        if self._channel_size != input_channel:
            images = tf.image.grayscale_to_rgb(images)
        return images, labels
