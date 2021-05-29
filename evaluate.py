import argparse
import os
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

from preprocess import TFImagePreprocessing


def main(
    dataset_name, 
    model_type,
    model_dir,
    batch_size,
):
    # Load test dataset
    test_dataset, info = tfds.load(
        name=dataset_name, 
        split="test", 
        with_info=True,
    )
    hight_size, width_size, channel_size = info.features["image"].shape
    num_classes = info.features["label"].num_classes

    # Image preprocess
    img_prep = TFImagePreprocessing(
        hight_size=hight_size, 
        width_size=width_size,
        channel_size=channel_size,
        num_classes=num_classes,
    )
    if model_type == "SimpleCNN":
        test_dataset = test_dataset.map(img_prep.base_preprocess)
        test_dataset = test_dataset.batch(batch_size)
    elif model_type == "VGG16":
        test_dataset = test_dataset.map(img_prep.vgg_preprocess)
        test_dataset = test_dataset.batch(batch_size)
    elif model_type == "Xception":
        test_dataset = test_dataset.map(img_prep.xception_preprocess)
        test_dataset = test_dataset.batch(batch_size)
    else:
        raise ValueError(f"The model: {model_type} does not exist.")

    # Load model
    model_file_name = model_type + ".h5"
    model_path = os.path.join(model_dir, model_file_name)
    model = load_model(model_path)

    # Evaluation
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"test_loss: {test_loss}, test_acc: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for evaluate task")
    parser.add_argument("--dataset_name", type=str, default="mnist")  # Ex.: mnist, fashion_mnist, cifar10
    parser.add_argument("--model_type", type=str, default="SimpleCNN")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )
