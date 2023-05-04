import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('digits.model')

for x in range(1, 6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    img = cv.resize(img, (28, 28))
    img = np.array(img).reshape(-1, 28, 28)
    img = tf.keras.utils.normalize(img, axis=1)
    prediction = model.predict(img)
    print(f'result is probably : {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
