import cv2 
from PIL import Image
from keras.models import load_model
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('pred\pred0.jpg')
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)

#print(img)
input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)
predicted_class = np.argmax(result, axis=-1)
print(predicted_class)
