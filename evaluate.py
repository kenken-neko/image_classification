import argparse
import os
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, smart_resize

from original_dataset_loader import OriginalDatasetLoader
from preprocess import TFImagePreprocessing


def main(
    dataset_name,
    original_dataset_path,
    single_image_height,
    single_image_width,
    single_image_path,
    model_type,
    model_dir,
):
    # Load test dataset
    if single_image_path:
        # Load and preprocess single image
        test_image = load_img(single_image_path)
        test_image = img_to_array(test_image) / 255
        test_image = smart_resize(
            test_image, 
            size=(single_image_height, single_image_width),
        )
        test_image = test_image[None, ...]
    elif dataset_name:
        test_dataset, info = tfds.load(
            name=dataset_name, 
            split="test", 
            with_info=True,
        )
        hight_size, width_size, channel_size = info.features["image"].shape
        num_classes = info.features["label"].num_classes
    elif original_dataset_path:
        # Load original dataset from directory
        ds_loader = OriginalDatasetLoader(
            original_dataset_path,
            valid_per_train=0,
        )
        (test_dataset, _), info = ds_loader.load()
        hight_size = info["hight_size"]
        width_size = info["width_size"]
        channel_size = info["channel_size"]
        num_classes = info["num_classes"]
    else:
        raise AssertionError("The dataset is not specified correctly.") 

    # Load model
    model_file_name = model_type + ".h5"
    model_path = os.path.join(model_dir, model_file_name)
    model = load_model(model_path)

    # Evaluation
    if single_image_path:
        if model_type == "SimpleCNN":
            target_size = None
        elif model_type == "VGG16":
            target_size = (224, 224)
        elif model_type == "Xception":
            target_size = (299, 299)
        else:
            raise ValueError(f"The model: {model_type} does not exist.")
        pred = model.predict(test_image, batch_size=1, verbose=0)
        pred_label = np.argmax(pred[0])
        score = np.max(pred)
        print(f"Predict label: {pred_label}")
        print(f"score: {score}")
    else:
        # Batch evaluation
        img_prep = TFImagePreprocessing(
            hight_size=hight_size, 
            width_size=width_size,
            channel_size=channel_size,
            num_classes=num_classes,
        )  # Image preprocess
        if model_type == "SimpleCNN":
            test_dataset = test_dataset.map(img_prep.base_preprocess)
        elif model_type == "VGG16":
            test_dataset = test_dataset.map(img_prep.vgg_preprocess)
        elif model_type == "Xception":
            test_dataset = test_dataset.map(img_prep.xception_preprocess)
        else:
            raise ValueError(f"The model: {model_type} does not exist.")
        # Batch
        test_dataset = test_dataset.batch(1)
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f"test_loss: {test_loss}, test_acc: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for evaluate task")
    parser.add_argument("--dataset_name", type=str, default=None)  # Ex.: mnist, fashion_mnist, cifar10
    parser.add_argument("--original_dataset_path", type=str, default=None)
    parser.add_argument("--single_image_path", type=str, default=None)
    parser.add_argument("--single_image_height", type=int, default=None)
    parser.add_argument("--single_image_width", type=int, default=None)
    parser.add_argument("--target_size", type=tuple, default=None)
    parser.add_argument("--model_type", type=str, default="SimpleCNN")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        original_dataset_path=args.original_dataset_path,
        single_image_path=args.single_image_path,
        single_image_height=args.single_image_height,
        single_image_width=args.single_image_width,
        model_type=args.model_type,
        model_dir=args.model_dir,
    )
