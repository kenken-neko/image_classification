import argparse
import os
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from models import SimpleCNNModel, VGG16Model, XceptionModel
from preprocess import TFImagePreprocessing


def main(
    dataset_name,
    is_augmentation,
    valid_per_train,
    model_type,
    is_fine_tuning,
    epochs,
    batch_size,
    optimizer,
    output_model_dir,
    log_dir,
):
    # Load train dataset
    train_ratio = int((1-valid_per_train)*100)
    (train_dataset, valid_dataset), info = tfds.load(
        name=dataset_name, 
        split=[
            f"train[:{train_ratio}%]",  # Train dataset 
            f"train[{train_ratio}%:]",  # Valid dataset
        ], 
        with_info=True,
    )
    hight_size, width_size, channel_size = info.features["image"].shape
    num_classes = info.features["label"].num_classes

    # Image preprocess instance
    img_prep = TFImagePreprocessing(
        hight_size=hight_size, 
        width_size=width_size,
        channel_size=channel_size,
        num_classes=num_classes,
    )
    if model_type == "SimpleCNN":
        # Image preprocess
        train_dataset = train_dataset.map(img_prep.base_preprocess)
        valid_dataset = valid_dataset.map(img_prep.base_preprocess)
        train_dataset = train_dataset.batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)
        # Build models
        model = SimpleCNNModel(
            hight_size=hight_size,
            width_size=width_size,
            channel_size=channel_size,
            num_classes=num_classes,
            is_augmentation=is_augmentation,
        )
    elif model_type == "VGG16":
        # Image preprocess
        train_dataset = train_dataset.map(img_prep.vgg_preprocess)
        valid_dataset = valid_dataset.map(img_prep.vgg_preprocess)
        train_dataset = train_dataset.batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)
        # Build models
        model = VGG16Model(
            num_classes=num_classes,
            is_fine_tuning=is_fine_tuning,
            is_augmentation=is_augmentation,
        )
    elif model_type == "Xception":
        # Image preprocess
        train_dataset = train_dataset.map(img_prep.xception_preprocess)
        valid_dataset = valid_dataset.map(img_prep.xception_preprocess)
        train_dataset = train_dataset.batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)
        # Build models
        model = XceptionModel(
            num_classes=num_classes,
            is_fine_tuning=is_fine_tuning,
            is_augmentation=is_augmentation,
        ) 
    else:
        raise ValueError(f"The model: {model_type} does not exist.")

    model = model.build()
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    model_file_name = model_type + ".h5"
    output_model_path = os.path.join(output_model_dir, model_file_name)
    # Preparing callbacks
    callbacks = [
        TensorBoard(log_dir=log_dir),
        EarlyStopping(patience=3),
        ModelCheckpoint(output_model_path)
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        batch_size=batch_size, 
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_dataset,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for train task")
    parser.add_argument("--dataset_name", type=str, default="mnist")  # Ex.: mnist, fashion_mnist, cifar10
    parser.add_argument("--is_augmentation", action="store_true")
    parser.add_argument("--valid_per_train", type=float, default=0.2)
    parser.add_argument("--model_type", type=str, default="SimpleCNN")
    parser.add_argument("--is_fine_tuning", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--output_model_dir", type=str, default="models")
    parser.add_argument("--log_dir", type=str, default="./tmp_logs/")
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        is_augmentation=args.is_augmentation,
        valid_per_train=args.valid_per_train,
        model_type=args.model_type,
        is_fine_tuning=args.is_fine_tuning,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        output_model_dir=args.output_model_dir,
        log_dir=args.log_dir,
    )
