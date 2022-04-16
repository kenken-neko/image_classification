import csv
import pathlib
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class OriginalDatasetLoader:
    def __init__(
        self,
        dataset_path,
        valid_per_train,
        hight_size=256,
        width_size=256,
        channel_size=3,
        output_label_dict_path="models/label_dict.csv",
    ):
        # Set image paths
        self._data_root = pathlib.Path(dataset_path)
        self._image_paths = list(self._data_root.glob("*/*"))
        self._image_paths = [str(path) for path in self._image_paths]
        # Set number of train images
        num_images = len(self._image_paths)
        self._num_train_images = int(num_images * (1 - valid_per_train))
        # Set channel size
        self._info = {}
        self._info.update({"hight_size": hight_size})
        self._info.update({"width_size": width_size})
        self._info.update({"channel_size": channel_size})
        # Set output path
        self._output_label_dict_path = output_label_dict_path

    def _preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=self._info["channel_size"])
        image = tf.image.resize(
            image,
            [self._info["hight_size"], self._info["width_size"]],
        )
        return image

    def _load_image_ds(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self._image_paths)
        image_ds = path_ds.map(
            self._preprocess_image,  # Image load
            num_parallel_calls=AUTOTUNE,
        )
        return image_ds

    def _load_label_ds(self):
        label_names = sorted(
            item.name for item in self._data_root.glob("*/") if item.is_dir()
        )
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        image_labels = [
            label_to_index[pathlib.Path(path).parent.name] for path in self._image_paths
        ]
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(image_labels, tf.int64))

        # For checking labels on inference results
        with open(self._output_label_dict_path, "w") as f:
            writer = csv.writer(f)
            for k, v in label_to_index.items():
                writer.writerow([k, v])

        # Get number of classes
        image_labels_set = set(image_labels)
        self._info.update({"num_classes": len(image_labels_set)})
        return label_ds

    def load(self):
        # Make image and label datasets
        image_ds = self._load_image_ds()
        label_ds = self._load_label_ds()
        # Zip
        zip_dataset = tf.data.Dataset.zip((image_ds, label_ds))
        # Split
        train_dataset = zip_dataset.shuffle(len(zip_dataset)).take(
            self._num_train_images
        )
        valid_dataset = zip_dataset.shuffle(len(zip_dataset)).skip(
            self._num_train_images
        )
        # Dict
        train_dataset = train_dataset.map(lambda x, y: {"image": x, "label": y})
        valid_dataset = valid_dataset.map(lambda x, y: {"image": x, "label": y})
        # PrefetchDataset
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)
        return (train_dataset, valid_dataset), self._info.copy()
