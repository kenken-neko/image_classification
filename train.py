import argparse
import os
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models import SimpleCNNModel, VGG16Model, XceptionModel
from preprocess import ImagePreprocess


def main(
    dataset_name,
    model_type,
    is_fine_tuning,
    epochs,
    batch_size,
    optimizer,
    output_model_path,
):
    # Load dataset
    train_dataset, info = tfds.load(
        name=dataset_name, 
        split="train", 
        with_info=True,
    )
    hight_size, width_size, channel_size = info.features["image"].shape
    num_classes = info.features["label"].num_classes

    # Image preprocess instance
    preprocess = ImagePreprocess(
        channel_size=channel_size,
        num_classes=num_classes,
    )

    # Build models
    if model_type == "SimpleCNN":
        model = SimpleCNNModel(
            hight_size=hight_size,
            width_size=width_size,
            channel_size=channel_size,
            num_classes=num_classes,
        )
        # Image preprocess
        train_dataset = train_dataset.map(preprocess.base_preprocess)
        train_dataset = train_dataset.batch(batch_size)
    elif model_type == "VGG16":
        model = VGG16Model(
            num_classes=num_classes,
            is_fine_tuning=is_fine_tuning,
        )
        # Image preprocess
        train_dataset = train_dataset.map(preprocess.vgg_preprocess)
        train_dataset = train_dataset.batch(batch_size)
    elif model_type == "Xception":
        model = XceptionModel(
            num_classes=num_classes,
            is_fine_tuning=is_fine_tuning,
        )
        # Image preprocess
        train_dataset = train_dataset.map(preprocess.xception_preprocess)
        train_dataset = train_dataset.batch(batch_size)    
    else:
        raise ValueError(f"The model: {model_type} does not exist.")

    model = model.build()
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    # Preparing callbacks
    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(output_model_path)
    ]

    # Train the model
    history = model.fit(
        train_dataset, 
        batch_size=batch_size, 
        epochs=epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for train task")
    parser.add_argument("--dataset_name", type=str, default="mnist")  # Ex.: mnist, fashion_mnist, cifar10
    parser.add_argument("--model_type", type=str, default="SimpleCNN")
    parser.add_argument("--is_fine_tuning", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--output_model_path", type=str, default="models/model.h5")
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        is_fine_tuning=args.is_fine_tuning,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        output_model_path=args.output_model_path,
    )
