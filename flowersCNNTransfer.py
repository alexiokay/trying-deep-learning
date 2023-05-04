
import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from sklearn.metrics import classification_report, confusion_matrix
#1. Install Dependencies and Setup
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

#2. Remove dodgy images

data_dir = 'data'
image_exts = ['jpeg','jpg', 'bmp', 'png']



# img = cv2.imread(os.path.join(data_dir, 'daisy', '5547758_eea9edfd54_n.jpg'))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            with Image.open(image_path) as img:
                tip = img.format.lower()

            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

#3. Load Images

batch_size = 92
img_height, img_width = 256, 256

flowers_images_dict = {}
data_dir = pathlib.Path("data")
# fill the dictionary with image paths
flowers_images_dict = {f.name: list(f.glob("*")) for f in data_dir.iterdir() if f.is_dir()}

flowers_labels = sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
flowers_labels_dict = {}
for i, subdir in enumerate(data_dir.iterdir()):
    if subdir.is_dir():
        flowers_labels_dict[subdir.name] = i

X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)






#5. Split data
# Split the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Split the train data into train and validation sets (80/20 split)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Print the shapes of the resulting datasets
print("Training set: ", X_train.shape, y_train.shape)
print("Validation set: ", X_valid.shape, y_valid.shape)
print("Test set: ", X_test.shape, y_test.shape)

#6. Preprocess Data
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2)
train_gen = train_datagen.flow(X_train_scaled, y_train, batch_size=96, shuffle=True)


plt.axis('off')
plt.imshow(train_gen[0][0][0])
plt.show()

#6. Deep Learning Model
classes_count = len(flowers_labels_dict)
print(classes_count)

vgg_model = tf.keras.applications.vgg16.VGG16()



# convert to Sequential model, omit the last layer
# this works with VGG16 because the structure is linear
model = tf.keras.models.Sequential()
for layer in vgg_model.layers[0:-1]:
    model.add(layer)

# set trainable=False for all layers
# we don't want to train them again
for layer in model.layers:
    layer.trainable = False

model.add(Dense(classes_count))

# loss and optimizer
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# get the preprocessing function of this model
preprocess_input = tf.keras.applications.vgg16.preprocess_input


model.summary()



#7. Train Model
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Define early stopping callbacks
early_stopping = EarlyStopping(monitor='loss', patience=4)
early_stopping_acc = EarlyStopping(monitor='val_accuracy', patience=4)


hist = model.fit(train_gen, epochs=10, validation_data=(X_valid, y_valid), callbacks=[tensorboard_callback, early_stopping, early_stopping_acc])

#8. Plot performance

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#9. Evaluate Model
model.evaluate(X_test_scaled,y_test)
predictions = model.predict(X_test_scaled)
score = tf.nn.softmax(predictions[0])
print(np.argmax(score))
print(y_test[0])

# print precision, recall and accuracy metrics
print(classification_report(y_test, np.argmax(predictions, axis=1), target_names=flowers_labels))


#10. Test Model
img = cv2.imread(os.path.join(data_dir, 'daisy', '5547758_eea9edfd54_n.jpg'))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


resize = tf.image.resize(img, (224,224))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))
yhat_class = flowers_labels[np.argmax(yhat)]

print('Predicted class: {}'.format(yhat_class))

# print the class with the highest probability from yhat
print(flowers_labels[np.argmax(yhat)])


#10. Save Model

model.save(os.path.join('models', 'imageCNN.h5'))
