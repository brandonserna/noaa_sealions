'''
Kaggle NOAA Stellar Sea Lions
'''
import keras
import cv2
import sys
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Input, concatenate
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16

# Configuration
n_classes = 5                   # amount of lion classes
batch_size = 8
epochs = 2                    # number of epochs (start @ 100) 150 better score
image_size = 512                # resized img
n_train_images = 948
n_test_images = 18636
model_name = input('Name of model run: ')
seed = 7
np.random.seed(seed)

train_dir = '/media/bss/Ubuntu HDD/noaa-sealions/data_512/train'
validation_dir = '/media/bss/Ubuntu HDD/noaa-sealions/data_512/validation'
test_dir = '/media/bss/Ubuntu HDD/noaa-sealions/data_512/test'
base_dir = '/media/bss/Ubuntu HDD/noaa-sealions/data_512/'

ignore_list = pd.read_csv('../data/miss_class.txt')['train_id'].tolist()
df = pd.read_csv('../data/train.csv')

# Tests
print('No tests configured...')

# Data prep
os.mkdir('./models/' + str(model_name))  # mk drop place for resources
cur_work_dir = os.path.join('./models/', model_name)
image_list = []  # store
y_list = []      # store
test_files = [i for i in os.listdir('../data_512/') if i.endswith('.png')]

for i in range(0, n_train_images):
    img_path = os.path.join('../data_512/all_data/', str(i) + '.png')
    img = cv2.imread(img_path)
    print('Image shape: ' + str(img.shape))

    image_list.append(img)

    row = df.ix[i]

    y_row = np.zeros((5))
    y_row[0] = row['adult_males']
    y_row[1] = row['subadult_males']
    y_row[2] = row['adult_females']
    y_row[3] = row['juveniles']
    y_row[4] = row['pups']
    y_list.append(y_row)
print('Images Loaded')
print('Y_list: ' + str(len(y_list)))
print('Image_list: ' + str(len(image_list)))
x_train = np.asarray(image_list)
y_train = np.asanyarray(y_list)

print('X Train: ' + str(x_train.shape))
print('Y Train: ' + str(y_train.shape))

# K Fold validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cv_scores = []  # store
#l, ct = y_train.shape
#labels = y_train.reshape(l,)

# model
vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(image_size,image_size,3))

# cross validation
for train, test in kfold.split(x_train, y_train[:,1]):
    # custom layers
    x= Conv2D(n_classes, (1, 1), activation='relu')(vgg16.output)
    x= GlobalAveragePooling2D()(x)
    model = Model(vgg16.input, x)

    print(model.summary())

    history = model.compile(loss=keras.losses.mean_squared_error,
            optimizer= keras.optimizers.Adadelta(), metrics=['accuracy'])
    # checkpointing
    file_p = cur_work_dir + '/' + 'best_weights.h5'
    checkpoint = ModelCheckpoint(file_p, monitor='loss', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]


    # Run
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=8), steps_per_epoch = len(x_train) / batch_size, epochs = epochs, callbacks=callback_list)

    # evaluate the model
    scores = model.evaluate(x_train[test], y_train[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cv_scores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
model.save(str(cur_work_dir) +'/' +  model_name + '.h5')


# submission
#model = load_model('./models/' + 'transvgg16_100e_wAug_wAdam.h5')
'''
test_files = [i for i in os.listdir('../data_512/') if i.endswith('.png')]

pred_arr = np.zeros((n_test_images, n_classes), np.int32)

for k in range(0, n_test_images):
    image_path = '../data_512/' + str(k) + '.png'

    img = cv2.imread(image_path)
    img = img[None, ...]
    pred = model.predict(img).astype(int)

    pred_arr[k,:] = pred

print('Pred arr: ' + str(pred_arr.shape))

pred_arr = pred_arr.clip(min=0)

df_submission = pd.DataFrame()
df_submission['test_id'] = range(0, n_test_images)
df_submission['adult_males'] = pred_arr[:,0]
df_submission['subadult_males'] = pred_arr[:,1]
df_submission['adult_females'] = pred_arr[:,2]
df_submission['juveniles'] = pred_arr[:,3]
df_submission['pups'] = pred_arr[:,4]

df_submission.to_csv(cur_work_dir + '/' + model_name + 'submission.csv', index = False)
'''
print('Complete')

# eval

print('\n\nEvaluation:')
print(history.history.keys())
