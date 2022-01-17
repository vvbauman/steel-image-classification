#%%
import os
import numpy as np
import pandas as pd
import image_functions 
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# define loc where augmented images should be saved
save_dir= 'C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full_aug3//'


# load the original images to be augmented
image_dir= 'C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full//all//' # loc of the 5 original images from the Severstal dataset
resize= 224
img_dataset= image_functions.img_files_to_array(image_dir,resize,5)

#%%
# define data augmentation
datagen= ImageDataGenerator(
    rotation_range= 40, 
    width_shift_range= 0.2, 
    height_shift_range= 0.3,
    zoom_range= 0.2, 
    horizontal_flip= True,
    brightness_range= [0.2,1.2] 
    )

#%%
# generate images
i= 0
for batch in datagen.flow(
    img_dataset,
    y= [1,0,2,3,4], # images in image_dir are alphabetically ordered. These are the corresponding image labels
    batch_size= 5,
    save_to_dir= save_dir,    
    save_prefix= 'aug',
    save_format= 'jpg',
    seed= i
    ):
        i += 1
        if i >1503: # returns 7010 images
            break

# %%
# create .csv with each image filename and its corresponding classification label
files= np.array(os.listdir(save_dir))
labels= []
for j in files:
    if j[4] == '0':
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
#np.savetxt('C:/Users/valerie.bauman/Documents/steel_defect_detection/augmented_images_3.csv', temp_arr, delimiter= ',', fmt= '%s')

# assign images to training/validation sets (90/10 split). Specify which set each image belongs to in the .csv
x_train, x_test, y_train, y_test= train_test_split(
    temp_arr[:,0],
    temp_arr[:,1],
    test_size= 0.1,
    random_state= 3,
    stratify= temp_arr[:,1]
    )

train_or_validation= []
for i in range(len(temp_arr)):
    if temp_arr[i] in np.array(x_train):
        train_or_validation.append('Train')
    else:
        train_or_validation.append('Validation')

# files_labels['Train or Validation'] = train_or_validation
# files_labels.to_csv('C:/Users/valerie.bauman/Documents/steel_defect_detection/augmented_images_3.csv', sep=',', index= False)


