# Image classification
This is an experimental script to perform image classification using Tensorflow's Keras and tensorflow_datasets.  
The following arguments can be used for learning or evaluation.  

# Arguments
train.py  
| Argument name | Description | Default | Data type |
| ------------- | ------------- | ------------- | ------------- |
| dataset_name | Dataset name for tensorflow_datasets. | None | str |
| original_dataset_path | Original dataset path. | None | str |
| dataset_size | Number of images in train dataset. If it is set to -1, all train image in tensorflow_datasets will be used. | -1 | int |
| augmentation_times | Increase the number of image by augmentation_times times by random augmentation. If set to 0, no image augmentation is performed. The augmentation process are horizontal flip and brightness adjustment.| 0 | int |
| augmentation_seed | Seed value for the random augmentation. | 0 | int |
| valid_per_train | Ratio of evaluation dataset to train dataset. | 0.2 | float |
| model_type | Model name for image classification. Specify one of the pre-defined "SimpleCNN", "VGG16", and "Xception". | SimpleCNN | str |
| is_fine_tuning | Specifies whether or not to perform fine-tuning. Fine-tuning is available only when model VGG16 or Xception is selected. | store_true | bool |
| is_dropout | Whether dropout layer is included or not. | store_true | bool |
| epochs | Number of training epochs. | 10 | int |
| batch_size | Size of training batch. | 32 | int |
| optimizer | Specify the optimization name in Keras optimizer. | adam | str |
| output_model_dir |  The name of output directory name for the trained model. | models | str |
| log_dir | The name of the output directory of the log to be used for tensorboard. | tmp_logs | str |

evaluate.py  
| Argument name | Description | Default | Data type |
| ------------- | ------------- | ------------- | ------------- |
| dataset_name | Dataset name for tensorflow_datasets. | None | str |
| original_dataset_path | Original dataset path. | None | str |
| single_image_path | Target single image path. | None | str |
| single_image_height | Height of target single image. | None | int |
| single_image_width | Width of target single image. | None | int |
| model_type | Model name for image classification. Specify one of the pre-defined "SimpleCNN", "VGG16", and "Xception". | SimpleCNN | str |
| model_dir |  The name of directory name for the trained model. | models | str |

# Sample docker commands
```
docker build . -t imageclass
docker run \
  -v $(pwd):/app \
  -it \
  --rm \
  imageclass \
  /bin/sh -c "python train.py --epochs 5"
```

# References
- [Image classification by Tensorflow Keres](https://www.tensorflow.org/tutorials/images/classification)
- [Load images by `tensorflow-dataset`](https://www.tensorflow.org/tutorials/load_data/images)
