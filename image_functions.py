#%%
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def img_files_to_array(dir, resize, n):
    """
    Reads in images and stores all in array
    Used to read in small dataset for image augmentation or to larger dataset for splitting into train/test sets

    Parameters
    ----------
    dir : str specifying the directory where the individual images to be read in are stored

    resize : int specifying the width and height the image should be resized to upon being read in. 
            Resizes images so they are square
            If images don't need to be resized, set this value to the image's current pixel width/height 

    n : int specifying the number of images in dir

    Returns
    ----------
    img_dataset : np array holding all resized images read in from dir
    """
    img_dataset= []
    imgs= os.listdir(dir)
    for i, image_name in enumerate(imgs):
        if(image_name.split('.')[1] == 'jpg'):
            image= Image.open(dir + image_name)
            #image= image.resize((resize, resize))
            img_dataset.append(np.array(image))
    #img_dataset= np.array(img_dataset).reshape(n, resize, resize, 3) # dimensions: (n_samples, height, width, n_channels)
    img_dataset= np.array(img_dataset)
    return img_dataset

def get_img_csv(aug_dir, save_dir):
    """
    After augmented images are created according to the naming convention used in generate_aug_img, use this function
    to create a csv where each row is 1 image and the columns are image file name and defect class label

    Parameters
    ----------
    aug_dir : str specifying the directory where all augmented images are stored

    save_dir : str specifying the csv file name and directory where the csv should be saved (e.g. 'C:/Users/valerie.bauman/Documents/steel_defect_detection/augmented_images.csv')

    Returns
    ----------
    None. (saves a csv as described above)

    """
    files= np.array(os.listdir(aug_dir))
    labels= []
    for j in files:
        if j[4] == '0': # first characters in all augmented images are "aug_" immediately followed by the image label (i.e. 0, 1, 2, 3, or 4)
            labels.append(0)
        elif j[4] == '1':
            labels.append(1)
        elif j[4] == '2':
            labels.append(2)
        elif j[4] == '3':
            labels.append(3)
        elif j[4] == '4':
            labels.append(4)
    temp_arr= np.array([files, labels]).T
    np.savetxt(save_dir, temp_arr, delimiter= ',', fmt= '%s')
    return

def train_val_split_aug(save_dir, val_split= 0.1, random_state= 3):
    """
    Assigns all augmented images to be part of either the training set or validation set

    Parameters
    ----------
    save_dir : str specifying the csv file name and directory where the csv with all augmented image file names
                and their labels are saved (e.g. 'C:/Users/valerie.bauman/Documents/steel_defect_detection/augmented_images.csv')

    val_split : float between 0 and 1 specifying the proportion of images that should be in the validation set

    random_state : int specifying the random seed to use for the train/validation split

    Returns
    ----------
    None. (overwrites the file in save_dir by adding a "Train or Validation" column)  

    """

    files_labels= pd.read_csv(save_dir, names= ['Filename', 'Label'], sep= ',')
    x_train, x_test, y_train, y_test= train_test_split(
        files_labels['Filename'],
        files_labels['Label'],
        test_size= val_split,
        random_state= random_state,
        stratify= files_labels['Label']
        )

    train_or_validation= []
    for i in range(len(files_labels)):
        if files_labels.iloc[i,0] in np.array(x_train):
            train_or_validation.append('Train')
        else:
            train_or_validation.append('Validation')

    files_labels['Train or Validation'] = train_or_validation
    files_labels.to_csv(save_dir, sep=',', index= False)
    return


def get_bottleneck_features(train_dir, target_size, train_batch_size, validation_dir, validation_batch_size, save_dir, train_bottleneck_name, validation_bottleneck_name, trained_model):
    """
    Get the bottleneck features for training and validation sets for ResNet50
    (i.e. output from pretrained ResNet50 model after training/validation data has made one pass)
    Save output, then use as the starting point to train a fully-connected model tacked onto the end of ResNet50
    Should only need to be run once, unless the ResNet50 model or batch sizes are changed

    Parameters
    ----------
    train_dir : str specifying the directory where training images are saved
                In this directory, there should be k sub-directories, where each sub-directory has the images belonging to the k-th class

    target_size : int specifying the width and height of the training and validation images (assumes images will be resized to be square)

    train_batch_size : int specifying the batch size to use for the training dataset

    validation_dir : str specifying the directory where validation images are saved
                    In this directory, there should be k sub-directories (same idea as train_dir)
    
    validation_batch_size: int specifying the batch size to use for the validation dataset

    save_dir : str specifying the directory where the model outputs should be saved
                Don't include a file name or extension

    train_bottleneck_name : str specifying the filename the model outputs will be saved to for the training set
                            Include extension .npy

    validation_bottleneck_name : str specifying the filename the model outputs will be saved to for the validation set
                                Include extension .npy

    trained_model : pre-trained Keras convolutional model

    Returns
    ----------
    nb_classes : int specifying the number of classes


    Also, ResNet50 output for the training and validation sets are saved as separate .npy files

    """

    # define ImageDataGenerator for the training and validation sets
    # assumes that images are augmented (i.e. all transformations have already been applied)
    data_generator= ImageDataGenerator(rescale= 1/255)

    # get bottleneck features for ResNet50 for the training set
    # i.e. run training data through ResNet50 once and save output, to eventually use as input to fully connected network
    bottleneck_gen= data_generator.flow_from_directory(
        train_dir,
        batch_size= train_batch_size,
        target_size= (target_size, target_size),
        class_mode= None, # don't predict image labels
        shuffle= False # don't need to shuffle since not training
    )
    nb_training= len(bottleneck_gen.filenames)
    nb_classes= len(bottleneck_gen.class_indices)
    train_pred_size= int(np.ceil(nb_training / train_batch_size))
    bottleneck_feats_train= trained_model.predict(bottleneck_gen, train_pred_size)
    np.save(open(os.path.join(save_dir + '\\' + train_bottleneck_name), 'wb'), bottleneck_feats_train)
    
    # repeat for validation set
    # lots of repeats here as above, so probably could be written in a better way, but fine for now
    bottleneck_gen= data_generator.flow_from_directory(
        validation_dir,
        batch_size= validation_batch_size,
        target_size= (target_size, target_size),
        class_mode= None,
        shuffle= False
    )
    nb_validation= len(bottleneck_gen.filenames)
    validation_pred_size= int(np.ceil(nb_validation / validation_batch_size))
    bottleneck_feats_valid= trained_model.predict(bottleneck_gen, validation_pred_size)
    np.save(open(os.path.join(save_dir + '\\' + validation_bottleneck_name), 'wb'), bottleneck_feats_valid)

    return nb_classes



# %%
