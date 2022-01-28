#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score
from PIL import Image
from image_functions import get_bottleneck_features
import tensorflow as tf
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
resize= 224 # original width is 400 pixels. Resize to be compatible with ResNet50 model
objective_function= 'binary_crossentropy' # Wang 2021 uses weighted cross entropy for their binary classifier. they also used focal loss
loss_metrics= ['accuracy']
batch_size_training= 32
batch_size_validation= 32
batch_size_testing= 128

# build pre-trained ResNet50 (can sub in for any other pre-trained convolutional model)
model= ResNet50(weights= 'imagenet', include_top= False)
model.trainable= False

# get bottleneck features for convolutional model
# (bottleneck features are input to fully connected model)
nb_classes= get_bottleneck_features(
    train_dir= train_dir,
    target_size= resize, 
    train_batch_size= batch_size_training,
    validation_dir= validation_dir,
    validation_batch_size= batch_size_validation,
    save_dir= os.getcwd(),
    train_bottleneck_name= 'bottleneck_features_train.npy',
    validation_bottleneck_name= 'bottleneck_features_validation.npy',
    trained_model= model
)

#%% 
# load bottleneck features 
train_data= np.load(open('bottleneck_features_train.npy', 'rb'))
train_labels= to_categorical(np.array([0] * 1266 + [1] * 5043))
validation_data= np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels= to_categorical(np.array([0] * 141 + [1] * 556))

# define and train a small fully connected model, using the bottleneck features as the starting point for training
top_model= Sequential()
top_model.add(Flatten(input_shape= train_data.shape[1:]))
top_model.add(Dense(512, activation= 'relu'))
top_model.add(Dense(128, activation= 'relu'))
top_model.add(Dense(2, activation= 'softmax'))

adam= optimizers.Adam(0.001)
top_model.compile(optimizer= adam, loss= objective_function, metrics= loss_metrics)

history= top_model.fit(
    train_data, train_labels,
    epochs= 60,
    batch_size= batch_size_training,
    validation_data= (validation_data, validation_labels)
    )

# save model weights, visualize learning curves
#top_model.save_weights('bottleneck_top.h5')
plt.plot(history.history['accuracy'], color= 'blue', label= 'Train acc')
plt.plot(history.history['val_accuracy'], color= 'orange', label= 'Validation acc')
plt.figure()
plt.plot(history.history['loss'], color= 'blue', label= 'Train loss')
plt.plot(history.history['val_loss'], color= 'orange', label= 'Validation loss')

#%%
# make predictions on test set
# create new ImageDataGenerator that scales pixel values 
data_generator= ImageDataGenerator(
    rescale= 1./255
)

test_gen= data_generator.flow_from_directory(
    test_dir, 
    target_size= (resize, resize),
    batch_size= 1, 
    class_mode= None, 
    shuffle = False 
    )
test_labels= np.array([0] * 941 + [1] * 6239)
bottleneck_pred= model.predict(test_gen)
final_pred_conf= top_model.predict(bottleneck_pred, batch_size= None)
final_pred= np.argmax(final_pred_conf, axis= 1)

# %%
print(confusion_matrix(test_labels, final_pred))
print(recall_score(test_labels, final_pred, average= 'micro'))
print(recall_score(test_labels, final_pred, average= None))


# %%
