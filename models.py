from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Lambda
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception


class SimpleCNNModel:

    def __init__(
        self, 
        hight_size,
        width_size,
        channel_size,
        num_classes,
        is_dropout=True,
    ):
        self._input = Input(shape=[hight_size, width_size, channel_size])
        self._conv2d = Conv2D(32, kernel_size=(3, 3), activation="relu")
        self._max_pooling = MaxPooling2D(pool_size=(2, 2))
        self._dropout = Dropout(0.25)
        self._flatten = Flatten()
        self._relu = Dense(128, activation="relu")
        self._softmax = Dense(num_classes, activation="softmax")
        self._is_dropout = is_dropout

    def build(self):
        input_data = self._input
        x = self._conv2d(input_data)
        x = self._max_pooling(x)
        if self._is_dropout:
            x = self._dropout(x)
        x = self._flatten(x)
        x = self._relu(x)
        output_data = self._softmax(x)
        return Model(inputs=[input_data], outputs=[output_data])


class VGG16Model:

    def __init__(
        self, 
        num_classes=None,
        is_fine_tuning=False,
    ):
        hight_size, width_size, channel_size = (224, 224, 3)
        self._input = Input(shape=[hight_size, width_size, channel_size])
        if is_fine_tuning:
            self._base_model = VGG16(weights="imagenet", include_top=False)
            for layer in self._base_model.layers[:-4]:
                layer.trainable = False
        else:
            self._base_model = VGG16(weights=None, include_top=False)
        self._flatten = Flatten(input_shape=(hight_size, width_size, channel_size))
        self._softmax = Dense(num_classes, activation="softmax")

    def build(self):
        input_data = self._input
        x = self._base_model(input_data)
        x = self._flatten(x)
        output_data = self._softmax(x)
        return Model(inputs=[input_data], outputs=[output_data])


class XceptionModel:

    def __init__(
        self, 
        num_classes=None,
        is_fine_tuning=False,
    ):
        hight_size, width_size, channel_size = (299, 299, 3)
        self._input = Input(shape=[hight_size, width_size, channel_size])
        if is_fine_tuning:
            self._base_model = Xception(weights="imagenet", include_top=False)
            for layer in self._base_model.layers[:-50]:
                layer.trainable = False
        else:
            self._base_model = Xception(weights=None, include_top=False)
        self._flatten = Flatten(input_shape=(hight_size, width_size, channel_size))
        self._softmax = Dense(num_classes, activation="softmax")

    def build(self):
        input_data = self._input
        x = self._base_model(input_data)
        x = self._flatten(x)
        output_data = self._softmax(x)
        return Model(inputs=[input_data], outputs=[output_data])
