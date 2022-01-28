#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import recall_score
from tensorflow.python.keras.layers.advanced_activations import Softmax
from image_functions import get_bottleneck_features
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

# global constants
train_dir= 'C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full_aug2_binary//train'
validation_dir= 'C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full_aug2_binary//validation'
test_dir= 'C://Users//valerie.bauman//Documents//steel_defect_detection//kaggle_train_full//one_defect//'
resize= 224 # original width is 400 pixels. Resize to be compatible with ResNet50
objective_function= 'binary_crossentropy' # Wang 2021 uses weighted cross entropy for their binary classifier. they also used focal loss
loss_metrics= ['accuracy']
batch_size_training= 64
batch_size_validation= 64
batch_size_testing= 128

# build pre-trained ResNet50 (can sub in for any other pre-trained convolutional model)
model= ResNet50(
    weights= 'imagenet',
    include_top= False)
model.trainable= False

# create new model to go on top of ResNet50
inputs= keras.Input(shape= (resize,resize,3))
x= model(inputs, training= False)
x= GlobalAveragePooling2D()(x)
intermed_outputs= Dense(128, activation= 'relu')(x)
outputs= Dense(2, activation= 'softmax')(intermed_outputs)
top_model= keras.Model(inputs,outputs)

#%%
# train top_model on new data (data runs through entire network in each epoch rather than just once, like what is done in the bottleneck features model)
# in ImageDataGenerator.flow_from_directory, image labels are automatically extracted 
data_generator= ImageDataGenerator(rescale= 1/255)
train_gen= data_generator.flow_from_directory(
        train_dir,
        batch_size= batch_size_training,
        target_size= (resize, resize)
    )
validation_gen= data_generator.flow_from_directory(
    validation_dir,
    batch_size= batch_size_validation,
    target_size= (resize, resize)
)

adam= optimizers.Adam(0.001)
top_model.compile(
    optimizer= adam,
    loss= objective_function,
    metrics= loss_metrics)
history= top_model.fit(
    train_gen,
    epochs= 50,
    validation_data= validation_gen
    )

#%%
# create new ImageDataGenerator that scales pixel values 
data_generator= ImageDataGenerator(
    rescale= 1./255,
)

test_gen= data_generator.flow_from_directory(
    test_dir, 
    target_size= (resize, resize),
    batch_size= 128, 
    class_mode= None, 
    shuffle = False 
    )
test_labels= np.array([0] * 941 + [1] * 6239) # specify labels of test set images
final_pred_conf= top_model.predict(test_gen)
final_pred= np.argmax(final_pred_conf, axis= 1)

print(recall_score(test_labels, final_pred, average= 'micro'))
print(recall_score(test_labels, final_pred, average= None))



# %%
