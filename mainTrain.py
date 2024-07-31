import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

image_directory = 'datasets/'

no_tumour_images =  os.listdir(image_directory+ 'no/')
yes_tumour_images =  os.listdir(image_directory+ 'yes/')
INPUT_SIZE=64
dataset=[]
label=[]
#print(no_tumour_images)

#path='no0.jpg'
#print(path.split('.')[1])

for i,image_name in enumerate(no_tumour_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)

for i,image_name in enumerate(yes_tumour_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)


dataset = np.array(dataset)
label = np.array(label)

x_train,x_test,y_train,y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)
#print(x_train.shape)  (2400,64,64,3)
#print(x_test.shape) #(600,64,64,3)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#Model Building
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3),  kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, 
batch_size=16, verbose=1, epochs=10, 
validation_data=(x_test,y_test),
shuffle=False
)

model.save('BrainTumor10EpochsCategorical.h5')



