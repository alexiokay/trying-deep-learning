# use tensorflow model import it
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import random

data_dir = 'data'
labels = sorted(os.listdir(data_dir))
print(labels)
#import model
model = load_model(os.path.join('models', 'imageCNN.h5'))

data_folder = 'data'
subfolders = os.listdir(data_folder)

# Choose a random subfolder
random_subfolder = os.path.join(data_folder, random.choice(subfolders))
print(random_subfolder)
# Get a list of all the images in the random subfolder
image_files = os.listdir(random_subfolder)

# Choose a random image
random_image_file = random.choice(image_files)

# Load the random image
img = cv2.imread(os.path.join(random_subfolder, random_image_file))


#resize image
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

#predict image

yhat = model.predict(np.expand_dims(resize/255, 0))
yhat_class = labels[np.argmax(yhat)]

print('Predicted class: {}'.format(yhat_class))

# print the class with the highest probability from yhat
print(labels[np.argmax(yhat)])

#--------


# augument image
img_augumented = tf.image.random_flip_left_right(resize)
img_augumented = tf.image.random_flip_up_down(img_augumented)
img_augumented = tf.image.random_brightness(img_augumented, 0.1)
img_augumented = tf.image.rot90(img_augumented, k=1)



#resize augumented image
resize = tf.image.resize(img_augumented, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
#predict augumented image
yhat = model.predict(np.expand_dims(resize/255, 0))
yhat_class = labels[np.argmax(yhat)]

print('Predicted augumented class: {}'.format(yhat_class))

