# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:03:47 2021

@author: asus123
"""

from glob import glob
import os
import pandas as pd
from skimage.io import imread
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
    
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import numpy as np
np.random.seed(2017)
def imread_size(in_path):
    t_img = imread(in_path)
    return zoom(t_img, [64/t_img.shape[0], 64/t_img.shape[1]]+([1] if len(t_img.shape)==3 else []),
               order = 2)


base_img_dir = os.path.join('./',)
all_training_images = glob(os.path.join(base_img_dir, '*', '*.png'))
full_df = pd.DataFrame(dict(path = all_training_images))
full_df['category'] = full_df['path'].map(lambda x: os.path.basename(os.path.dirname(x)))
full_df = full_df.query('category != "valid"')
cat_enc = LabelEncoder()
full_df['category_id'] = cat_enc.fit_transform(full_df['category'])
y_labels = to_categorical(np.stack(full_df['category_id'].values,0))
print(y_labels.shape)
full_df['image'] = full_df['path'].map(imread_size)
full_df.sample(3)

full_df['category'].value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(np.stack(full_df['image'].values,0),-1), 
                                                    y_labels,
                                   random_state = 12345,
                                   train_size = 0.75,
                                   stratify = full_df['category'])
print('Training Size', X_train.shape)
print(y_test)

from keras.preprocessing.image import ImageDataGenerator # (docu: https://keras.io/preprocessing/image/)

train_datagen = ImageDataGenerator(
        samplewise_std_normalization = True,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range = 360,
        )

test_datagen = ImageDataGenerator(
        samplewise_std_normalization = True)

train_gen = train_datagen.flow(X_train, y_train, batch_size=32)
test_gen = train_datagen.flow(X_test, y_test, batch_size=32)

fig, (ax1, ax2) = plt.subplots(2, 4, figsize = (12, 6))
for c_ax1, c_ax2, (train_img, _), (test_img, _) in zip(ax1, ax2, train_gen, test_gen):
    c_ax1.imshow(train_img[0,:,:,0])
    c_ax1.set_title('Train Image')
    
    c_ax2.imshow(test_img[0,:,:,0])
    c_ax2.set_title('Test Image')
    
from keras.models import Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet
mn_cnn = MobileNet(input_shape = train_img.shape[1:], dropout = 0.25, weights = None, 
                  classes = y_labels.shape[1])
mn_cnn.compile(loss = 'categorical_crossentropy', 
               optimizer = Adam(lr = 1e-4, decay = 1e-6),
               metrics = ['acc'])
loss_history = []
#mn_cnn.summary()

for i in range(20):
    loss_history += [mn_cnn.fit_generator(train_gen, steps_per_epoch=10,
                         validation_data=test_gen, validation_steps=10)]
    
    
epoch = np.cumsum(np.concatenate(
    [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
_ = ax1.plot(epoch,
             np.concatenate([mh.history['loss'] for mh in loss_history]),
             'b-',
             epoch, np.concatenate(
        [mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax2.plot(epoch, np.concatenate(
    [mh.history['acc'] for mh in loss_history]), 'b-',
                 epoch, np.concatenate(
        [mh.history['val_acc'] for mh in loss_history]),
                 'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('Accuracy')

##############################################################################################
import cv2
import numpy as np
from keras import preprocessing
image_path = "valid/DSC08320.jpg"

img = preprocessing.image.load_img(image_path, target_size=(64,64))
x = preprocessing.image.img_to_array(img)[:,:,1]

x = np.expand_dims(x, axis=2)
x = np.expand_dims(x, axis=0)
# print(x.shape)

out = mn_cnn.predict(x)
# print(out)

image = cv2.imread(r"valid\photos2\DSC08341.jpg")


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(11,11),0)


edged = cv2.Canny(blurred, 30, 150)


(cnts,_) = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("There are {} protential coins in the image".format(len(cnts)))


#裁剪
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0,255,0), 2)

extracted_circle = []
for (i,circle) in enumerate(cnts):
    # circle是每个提取出来的圆
    (x,y,w,h) = cv2.boundingRect(circle)
    print("coin #{}".format(i+1))
    coin_canvas = image[y:y+h, x:x+w]
    extracted_circle.append(coin_canvas)

# Test the fack circle 
def check(c):
    x,y,z = c.shape
    if x*y<1000:
        return False
    
    if x/y>0.8 and y/x>0.8:
        return True
    else:
        return False

extracted_coins=[]
for c in extracted_circle:
    if check(c):
        extracted_coins.append(c)

print("After checking, There are {} coins in the image".format(len(extracted_coins)))

dic = {i: 0 for i in range(7)}

for img in extracted_coins:
    x = np.array(img)[:,:,1]
    x = x.astype(np.float64)
    x = np.resize(x, (64,64))
    
    x = np.expand_dims(x, axis=2)
    x = np.expand_dims(x, axis=0)
    out = mn_cnn.predict(x)
    dic[np.argmax(out)]=dic[np.argmax(out)]+1

coin_dic = {0:'1fr', 1:'2fr', 2:'10rp', 3:'20rp', 4:'50rp', 5:'back_fr', 6:'back_rp'}

for key, value in dic:
    print('There are {} {} in the image. '.format(value, coin_dic[key]))