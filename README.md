# steel-image-classification

This repo includes code and images for defect classification in steel parts. The goal of this work was to classify (and eventually get the bounding boxes for) surface defects in steel using machine learning methods and only 1 image from each defect class. 1 image from each of the 5 defect classes in the [Severstal steel defect detection dataset](https://www.kaggle.com/c/severstal-steel-defect-detection) were used to generate the training and validation sets. The entire Severstal dataset was used as the test set. There are images labelled as defect/non-defect to enable binary classification and there are images labelled as one of the 5 defect classes to enable multi-class classification.

To use the data and scripts, I suggest cloning the repo to your computer or other cloud-based storage system. Any scripts that use directory strings will need to be modified.

## File descriptions
### Folders w/images
- train_colour_full_aug2 : folder. Includes training and validation images generated through image augmentation techniques of the 5 original images from the Severstal dataset. Sub-folders within this folder are set up for multi-class classification and to use keras' ImageDataGenerator.flow_from_directory()
- train_colour_full_aug2_binary : folder. Includes training and validation images generated through image augmentation techniques of the 5 original images from the Severstal dataset. Sub-folders within this folder are set up for binary classification and to use keras' ImageDataGenerator.flow_from_directory()
- train_colour_full : folder. Includes the original 5 images from the Severstal dataset that were used to generate the training/validation images in train_colour_full_aug2_binary. Filenames are unchanged from the original Severstal dataset

### Model scripts


### Processing scripts 
Note: these .py files were used to organize/rename the images in the folders described above
- generate_images_augmentation.py : script to generate the augmented images (for the training/validation sets) from the 5 original images from the Severstal dataset. This script uses images in train_colour_full to generate the images in train_colour_full_aug2_binary
- process_kaggle_imgs.py : script to process the entire Severstal training set. This script was used to rename/organize the images in the kaggle_train_full folder 



