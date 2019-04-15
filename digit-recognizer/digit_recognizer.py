import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import  LinearSVC
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout


def parse_img(img_array, size= 28):
    out = []
    for img_nbr in range(img_array.shape[0]):
        out.append(img_array[img_nbr, :].reshape(size, size, 1))

    return np.array(out)

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

labels = train_dataset.values[:, 0]
labels = np.ravel(labels)
train_images = train_dataset.values[:, 1:]
test_images = test_dataset.values[:, :]

train_images = parse_img(train_images)
test_images = parse_img(test_images)

# for i in range(10):
#     plt.imshow(random.choice(test_images))
#     plt.show()

# model = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]), labels)
# predicted_test = model.predict(test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]))

img_rows, img_cols = 28, 28
num_classes = 10

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_images, keras.utils.to_categorical(labels),
          batch_size=128,
          epochs=10,
          validation_split = 0.2)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predicted_test = model.predict(test_images)

predicted_label = np.argmax(predicted_test, axis= 1)#.reshape(predicted_test.shape[0],)
output = pd.DataFrame({'ImageId': list(range(1, test_images.shape[0]+1)), 'Label': predicted_label})
output.to_csv('prediction_cnn1.csv', index=False)


test = 1