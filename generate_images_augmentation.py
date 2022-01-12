#%%
import os
import numpy as np
import image_functions 
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    y= [1,0,2,3,4],
    batch_size= 5,
    save_to_dir= 'C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full_aug3',    
    save_prefix= 'aug',
    save_format= 'jpg',
    seed= i
    ):
        i += 1
        if i >1503: # returns 7010 images
            break

# %%
