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
        vgg_input_size = [224, 224]
        images, labels = self.base_preprocess(dataset)
        if self._image_size != vgg_input_size:
            images = tf.image.resize(
                images, vgg_input_size
            )
        if self._channel_size != 3:
            images = tf.image.grayscale_to_rgb(images)
        return images, labels

    def xception_preprocess(self, dataset):
        xcptn_input_size = [299, 299]
        images, labels = self.base_preprocess(dataset)
        if self._image_size != xcptn_input_size:
            images = tf.image.resize(
                images, xcptn_input_size
            )
        if self._channel_size != 3:
            images = tf.image.grayscale_to_rgb(images)
        return images, labels
