# steel-image-classification

This repo includes code and images for defect classification in steel parts. The goal of this work was to classify (and eventually get the bounding boxes for) surface defects in steel using machine learning methods and only 1 image from each defect class. 1 image from each of the 5 defect classes in the [Severstal steel defect detection dataset](https://www.kaggle.com/c/severstal-steel-defect-detection) were used to generate the training and validation sets. The entire Severstal dataset was used as the test set. The images were labelled as defect/non-defect, to enable binary classification.

## File descriptions

- train_color_full_aug2_binary : folder. Includes training and validation images generated through image augmentation techniques of the 5 original images from the Severstal dataset. Sub-folders within this folder are set up to use keras' ImageDataGenerator.flow_from_directory()
- generate_images_augmentation.py : script to generate the augmented images (for the training/validation sets) from the 5 original images from the Severstal dataset
