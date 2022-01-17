#%%
import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image
import image_functions

# this script has some of the processing of the full Severstal training set provided on Kaggle
# this script hasn't been touched since the Kaggle images were originally processed, so this code should be carefully inspected and tested before use

# %%
# get image labels for all images in Kaggle dataset
df= pd.read_csv('C:/Users/valerie.bauman/Documents/steel_defect_detection/train.csv') # dataframe that has all Kaggle images, including the filenames, defect types, and defect bounding boxes
df= df.drop_duplicates(subset= 'ImageId', keep= False) # drop images with more than one defect
img_id= df.iloc[:,0].values
labels= df.iloc[:,1].values

#for i in img_id:
#    shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/" + i), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/multiple_defects/" + i)

for j in [1,2,3,4]:
    for i in range(len(labels)):
        if labels[i] == j:
            shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/" + img_id[i]), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/label" + str(j) + "_" + img_id[i])



# %%
# rename images in kaggle dataset so the image label is in the file name
loc_kaggle= 'C://Users//valerie.bauman//Documents//steel_defect_detection//kaggle_train_full//train_images//'
files= np.array(os.listdir(loc_kaggle))
count1= 0
count2= 0
count3= 0
count4= 0
for i in files:
    if i[5] == '1':
        if i[6] == '_':
            count1 += 1
        else:
            shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/" + i), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/label0_" + i)        
    elif i[5] == '2':
        if i[6] == '_':
            count2 += 1
        else:
            shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/" + i), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/label0_" + i)        
    elif i[5] == '3':
        if i[6] == '_':
            count3 += 1
        else:
            shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/" + i), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/label0_" + i)        
    elif i[5] == '4':
        if i[6] == '_':
            count4 += 1
        else:
            shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/" + i), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/label0_" + i)        
#    else:
#        shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/" + i), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/train_images/label0_" + i)        

# %%
# get similarity between two images via cosine similarity and Euclidean distance
from scipy.spatial.distance import euclidean, cosine
test_img= np.array(os.listdir('C://Users//valerie.bauman//Documents//steel_defect_detection//kaggle_train_full//one_defect//all_images'))
train= np.array(Image.open('C://Users//valerie.bauman//Documents//steel_defect_detection//train_colour_full//all//00031f466.jpg')).flatten()/255
class0= []
for i in test_img:
    if i[5] == '0':
        if i[6] == '_':
            class0.append(i)

euc_dist= []
cosine_sim= []
for i in class0:
    temp_img= np.array(Image.open('C://Users//valerie.bauman//Documents//steel_defect_detection//kaggle_train_full//one_defect//all_images//' + i)).flatten()/255
    euc_dist.append(euclidean(train, temp_img))
    cosine_sim.append(cosine(train, temp_img))

sim_results= np.array([class0, euc_dist, cosine_sim])

# %%
# remove class 0 images from the test set that have a Euclidean distance >100 from the class 0 image used to make the training set
df= pd.read_csv('C:/Users/valerie.bauman/Documents/steel_defect_detection/class0_similarities.csv')
df= df.loc[df['Euclidean distance between image and training set image'] > 100]
img= df.iloc[:,0].values
for i in img:
    shutil.move(os.path.join("C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/one_defect/all_images/" + i), "C:/Users/valerie.bauman/Documents/steel_defect_detection/kaggle_train_full/dissimilar_class0" + i)        


# %%
