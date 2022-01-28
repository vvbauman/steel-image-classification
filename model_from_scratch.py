#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size= 224
batch_size_training= 64
batch_size_validation= 64
train_dir= 'C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full_aug2_binary//train'
validation_dir = 'C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full_aug2_binary//validation'

# define generators for training and validation data
data_generator= ImageDataGenerator(rescale= 1/255)
train_gen= data_generator.flow_from_directory(
    train_dir,
    batch_size= batch_size_training,
    target_size= (img_size, img_size),
    class_mode= 'categorical',
    shuffle= True 
)

validation_gen= data_generator.flow_from_directory(
    validation_dir,
    batch_size= batch_size_validation,
    target_size= (img_size, img_size),
    class_mode= 'categorical',
    shuffle= True
)

# define model
input_shape= (224, 224, 3)
input_tensor= Input(input_shape)
conv_1= Conv2D(16, (3,3), activation= 'relu', padding= 'same')(input_tensor)
conv_2= Conv2D(64, (3,3), activation= 'relu', padding= 'same')(conv_1)
conv_3= Conv2D(128, (3,3), activation= 'relu', padding= 'same')(conv_2)
flat= Flatten()(conv_3)
dense_1= Dense(128, activation= 'relu')(flat)
output= Dense(2, activation= 'softmax')(dense_1)
model= Model(input_tensor, output)
adam= optimizers.Adam(0.001)
model.compile(optimizer= adam, loss= 'binary_crossentropy', metrics= 'accuracy')

# train model
history= model.fit(
    x= train_gen,
    y= None,
    batch_size= None,
    epochs= 40,
    validation_data= validation_gen
)
plt.plot(history.history['accuracy'], color= 'blue', label= 'Train acc')
plt.plot(history.history['val_accuracy'], color= 'orange', label= 'Validation acc')
plt.figure()
plt.plot(history.history['loss'], color= 'blue', label= 'Train loss')
plt.plot(history.history['val_loss'], color= 'orange', label= 'Validation loss')

# print(model.summary())