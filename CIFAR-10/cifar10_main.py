import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import  LinearSVC
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, MaxPooling2D
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# def parse_img(img_array, size= 32):
#     out = []
#     for img_nbr in range(img_array.shape[0]):
#         r = img_array[img_nbr, 0:size*size].reshape(size, size)
#         g = img_array[img_nbr, size*size:2*size*size].reshape(size, size)
#         b = img_array[img_nbr, 2*size*size:3*size*size].reshape(size, size)
#
#         out.append(np.stack((r, g, b), axis=2))
#
#     return np.array(out)
#
# batch_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
# batches = []
# for batch in batch_names:
#     batches.append(unpickle("cifar-10-batches-py/" + batch))
#
# batches_data = []
# labels = []
# for batch in batches:
#     batches_data.append(parse_img(batch[b'data']))
#     labels.append(batch[b'labels'])
#
# test_batch = batches_data[5]
# test_labels = labels[5]
#
# """Uncomment to see random images, used to debug and make sure parsing was done correctly"""
# # for i in range(10):
# #     plt.imshow(random.choice(train_images))
# #     plt.show()

# """More complex CNN model"""
# img_rows, img_cols = 32, 32
# num_chan = 3
# num_classes = 10
# epochs = 100
#
# model = Sequential()
# model.add(Conv2D(20, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(img_rows, img_cols, num_chan)))
# model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(batches_data[0], keras.utils.to_categorical(labels[0]),
#           batch_size=128,
#           epochs=10,
#           validation_split = 0.2)
#
# """Print training history"""
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# """Predict and print to file"""
# predicted_test = model.predict(test_batch)
# predicted_label = np.argmax(predicted_test, axis= 1)









import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = False
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
shuffle=True)

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
    workers=4)

predicted_test = model.predict(x_test, verbose=1)
predicted_label = np.argmax(predicted_test, axis= 1)


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_label))

print(confusion_matrix(y_test, predicted_label))
# precision_score(test_labels, predicted_label, average = "macro")
# recall_score(test_labels, predicted_label, average = "macro")
# accuracy_score(test_labels, predicted_label)







