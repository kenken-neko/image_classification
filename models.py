from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception


class SimpleCNNModel:
    description = """ 
        This is a simple CNN model of my own creation.
        Init args:
            - hight_size      : Input image hight sise
            - width_size      : Input image width sise
            - channel_size    : Input image channel sise
            - num_classes     : Number of classes in the image dataset. 
                                Used for the softmax parameter.
    """
    def __init__(
        self, 
        hight_size,
        width_size,
        channel_size,
        num_classes,
        is_dropout=False,
    ):
        self._input = Input(shape=[hight_size, width_size, channel_size])
        self._conv2d = Conv2D(32, kernel_size=(3, 3), activation="relu")
        self._max_pooling = MaxPooling2D(pool_size=(2, 2))
        self._flatten = Flatten()
        self._relu = Dense(128, activation="relu")
        self._softmax = Dense(num_classes, activation="softmax")
        # Dropout function
        self._dropout = None
        if is_dropout:
            self._dropout = Dropout(0.25)

    def build(self):
        x = self._input
        x = self._conv2d(x)
        x = self._max_pooling(x)
        if self._dropout:
            x = self._dropout(x)
        x = self._flatten(x)
        x = self._relu(x)
        output_data = self._softmax(x)
        return Model(inputs=[self._input], outputs=[output_data])


class VGG16Model:
    description = """ 
        This is a VGG16 model that can use weights pre-trained in ImageNet.
        Init args:
            - num_classes     : Number of classes in the image dataset. 
                                Used for the softmax parameter.
            - is_fine_tuning  : Parameter to set whether or not to perform fine-tuning.
    """
    def __init__(
        self, 
        num_classes=None,
        is_fine_tuning=False,
        is_dropout=True,
    ):
        hight_size, width_size, channel_size = (224, 224, 3)
        self._input = Input(shape=[hight_size, width_size, channel_size])
        if is_fine_tuning:
            self._base_model = VGG16(weights="imagenet", include_top=False)
            for layer in self._base_model.layers[:-4]:
                layer.trainable = False
        else:
            self._base_model = VGG16(weights=None, include_top=False)
        self._average_pooling = GlobalAveragePooling2D()
        self._relu = Dense(512, activation="relu")
        self._dropout = None
        if is_dropout:
            self._dropout = Dropout(0.25)
        self._softmax = Dense(num_classes, activation="softmax")

    def build(self):
        x = self._input
        x = self._base_model(x)
        x = self._average_pooling(x)
        x = self._relu(x)
        if self._dropout:
            x = self._dropout(x)
        output_data = self._softmax(x)
        return Model(inputs=self._input, outputs=output_data)


class XceptionModel:
    description = """ 
        This is a Xception V1 model that can use weights pre-trained in ImageNet.
        Init args:
            - num_classes     : Number of classes in the image dataset. 
                                Used for the softmax parameter.
            - is_fine_tuning  : Parameter to set whether or not to perform fine-tuning.
    """
    def __init__(
        self, 
        num_classes=None,
        is_fine_tuning=False,
        is_dropout=True,
    ):
        hight_size, width_size, channel_size = (299, 299, 3)
        self._input = Input(shape=[hight_size, width_size, channel_size])
        if is_fine_tuning:
            self._base_model = Xception(weights="imagenet", include_top=False)
            for layer in self._base_model.layers[:-50]:
                layer.trainable = False
        else:
            self._base_model = Xception(weights=None, include_top=False)
        self._average_pooling = GlobalAveragePooling2D()
        self._relu = Dense(512, activation="relu")
        self._dropout = None
        if is_dropout:
            self._dropout = Dropout(0.25)
        self._softmax = Dense(num_classes, activation="softmax")

    def build(self):
        x = self._input
        x = self._base_model(x)
        x = self._average_pooling(x)
        x = self._relu(x)
        if self._dropout:
            x = self._dropout(x)
        output_data = self._softmax(x)
        return Model(inputs=self._input, outputs=output_data)
