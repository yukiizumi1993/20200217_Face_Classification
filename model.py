
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

home = os.getcwd()
input_dir = ["Aiba", "Matsumoto", "Ninomiya", "Ohno", "Sakurai"]

IMAGE_SIZE = 100
N_CATEGORIES = 5
BATCH_SIZE = 50
EPOCHS = 50
LEARNING_RATE = 0.0001

# Data extension
'''
for dir in input_dir:
    files = glob.glob(home + "/" + dir + "/*")
    output_dir = dir + "_extended"

    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    for i, file in enumerate(files):

        img = load_img(file)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        datagen = ImageDataGenerator(
            channel_shift_range=100,
            horizontal_flip=True,
        )
        g = datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='img', save_format='jpg')
        for i in range(4):
            batch = g.next()
'''

# Reading the Data
X = []
Y = []
for i, dir in enumerate(input_dir):
    obj = glob.glob(home + "/" + dir + "_extended" + "/*")
    for picture in obj:
        img = img_to_array(load_img(picture, target_size=(IMAGE_SIZE, IMAGE_SIZE)))
        X.append(img)
        Y.append(i)

# Change to array
X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape)

# Normalize the pixel value from 0 to 1
X = X.astype('float32')
X = X / 255.0

# One hot encoding
Y = np_utils.to_categorical(Y, N_CATEGORIES)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Learning
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(N_CATEGORIES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])
history = model.fit(X_train, y_train, verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
model.save("model_arashi_classification.h5", include_optimizer=False)
