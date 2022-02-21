import pathlib
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class OriginalDatasetLoader:

    def __init__(self, dataset_path, valid_per_train):
        # Set image paths
        self._data_root = pathlib.Path(dataset_path)
        self._image_paths = list(self._data_root.glob('*/*'))
        self._image_paths = [str(path) for path in self._image_paths]
        # Set number of train images
        num_images = len(self._image_paths)
        self._num_train_images = int(num_images * valid_per_train)

    def _preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def _load_image_ds(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self._image_paths)
        image_ds = path_ds.map(
            self._preprocess_image,  # Image load
            num_parallel_calls=AUTOTUNE,
        )
        train_image_ds = image_ds.take(self._num_train_images) 
        valid_image_ds = image_ds.skip(self._num_train_images)
        return train_image_ds, valid_image_ds

    def _load_label_ds(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self._image_paths)
        label_names = sorted(
            item.name for item in self._data_root.glob('*/') if item.is_dir()
        )
        label_to_index = dict(
            (name, index) for index, name in enumerate(label_names)
        )
        image_labels = [
            label_to_index[pathlib.Path(path).parent.name]
            for path in self._image_paths
        ]
        label_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast(image_labels, tf.int64)
        )
        train_label_ds = label_ds.take(self._num_train_images) 
        valid_label_ds = label_ds.skip(self._num_train_images) 
        return train_label_ds, valid_label_ds

    def load(self):
        train_image_ds, valid_image_ds = self._load_image_ds()
        train_label_ds, valid_label_ds = self._load_label_ds()
        train_dataset = tf.data.Dataset.zip((train_image_ds, train_label_ds))
        valid_dataset = tf.data.Dataset.zip((valid_image_ds, valid_label_ds))
        train_dataset = train_dataset.prefetch(1)
        valid_dataset = valid_dataset.prefetch(1)
        return train_dataset, valid_dataset
