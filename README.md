# steel-image-classification

This repo includes code and images for defect classification in steel parts. The goal of this work was to classify (and eventually get the bounding boxes for) surface defects in steel using machine learning methods and only 1 image from each defect class. 1 image from each of the 5 defect classes in the [Severstal steel defect detection dataset](https://www.kaggle.com/c/severstal-steel-defect-detection) were used to generate the training and validation sets. The entire Severstal dataset was used as the test set with some images from non-defect class removed. There are images labelled as defect/non-defect to enable binary classification and there are images labelled as one of the 5 defect classes to enable multi-class classification.

To use the data and scripts, I suggest cloning the repo to your computer or other cloud-based storage system. Any scripts that use directory strings will need to be modified.

## File descriptions
### Folders w/images
- train_colour_full_aug2 : folder. Includes training and validation images (7010 total) generated through image augmentation techniques of the 5 original images from the Severstal dataset. Sub-folders within this folder are set up for multi-class classification and to use keras' ImageDataGenerator.flow_from_directory()
- train_colour_full_aug2_binary : folder. Includes training and validation images (7010 total) generated through image augmentation techniques of the 5 original images from the Severstal dataset. Sub-folders within this folder are set up for binary classification and to use keras' ImageDataGenerator.flow_from_directory()
- train_colour_full : folder. Includes the original 5 images from the Severstal dataset that were used to generate the training/validation images in train_colour_full_aug2_binary. Filenames are unchanged from the original Severstal dataset
- kaggle_train_full : folder. Includes all images in the Severstal dataset (to be used as the test set for the defect classification models). Sub-folders include:
  - dissimilar_class0 : contains class0 (non-defect) images that are most dissimilar from the 1 class0 non-defect image used to create the training and validation sets. Dissimilar class0 images were those with a Euclidean distance greater than 100 between the image and the class0 image used to create the training and validation sets.
  - multiple_defects : contains the Severstal images that had multiple defects (out of scope, not used in this repo)
  - one_defect : contains all Severstal images that had only 1 or no defect, with the images in the dissimilar_class0 folder removed. All image names start with the image label

### Model scripts
- transfer_learning_bottleneck_features_binary.py : script to train and evaluate a transfer learning  model using a "bottleneck feature" approach. Model is a binary classifier (defect/non-defect) based on ResNet50. Training and validation images intended to be used are in the train_colour_full_aug2_binary folder. Test images are in kaggle_train_full folder (one_defect sub-folder)
- transfer_learning_bottleneck_features_multiclass.py : script to train and evaluate a transfer learning model using a "bottleneck feature" approach. Model is a multi-class classifier (5 defect classes) based on ResNet50. Training and validation images intended to be used are in the train_colour_full_aug2 folder. Test images are in the kaggle_train_full folder (one_defect sub-folder)
- transfer_learning_model_freeze_approach.py : script to train and evaluate a transfer learning model using a "model freeze" approach. Model is a binary classifier (defect/non-defect) based on ResNet50. Training and validation images intended to be used are in the train_colour_full_aug2_binary folder. Test images are in kaggle_train_full folder (one_defect sub-folder)
- model_from_scratch.py : script to train and evaluate a binary classifier (defect/non-defect) using a CNN with simple architecture. Training and validation images intended to be used are in the train_colour_full_aug2_binary folder.

*Keras blog describing the difference between the "bottleneck features" and "freeze model" transfer learning approaches: https://keras.io/guides/transfer_learning/.*

### Processing scripts 
Note: these .py files were used to organize/rename the images in the folders described above
- generate_images_augmentation.py : script to generate the augmented images (for the training/validation sets) from the 5 original images from the Severstal dataset. This script uses images in train_colour_full to generate the images in train_colour_full_aug2_binary
- process_kaggle_imgs.py : script to process the entire Severstal training set. This script was used to rename/organize the images in the kaggle_train_full folder 
- image_functions.py : functions used in model scripts. Functions include inline descriptions and comments
- train.csv : csv provided in the Severstal steel defect detection Kaggle competition. Provides image metadata, including each image's filename, defect class, and bounding boxes. Details on non-defect images are not included
- class0_similarities.csv : csv with the Euclidean distance and cosine similarity between each class0 (non-defect) image in the Severstal dataset and the 1 class0 image used to make the training and validation sets
