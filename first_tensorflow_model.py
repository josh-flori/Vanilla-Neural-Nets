# The purpose of this script is to create a tensorflow model on the same data as my hand-crafted network.
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import pandas as pd
from sklearn.model_selection import train_test_split
print(tf.__version__)


##################
#      PATH      #
##################
path='/users/josh.flori/desktop/images/'
m=len([i for i in os.listdir(path) if 'jpg' in i])


##################
#    load x/y    #
##################
X = np.array([imageio.imread(path+str(i)+'.jpg') for i in range(1,m+1)])          # So things to note here... mostly just... os.listdir returns some .DS_Store bullshit, so we sorted the list and it pops out in front...  then we get everything BUT that. It reads the matrix the wrong way so we flip it so that each column is an entire image, which I think is what we want. # 208 must be the total count of images + 1 for python to do it's thing
Y=np.array(pd.read_csv('/users/josh.flori/desktop/Y.csv')['Y'].values.tolist())
X.shape       # X.shape[1], Y.shape[1] should both be equal to m, or the number of training examples
Y.shape


##########################
#    STANDARDIZE DATA    #
##########################
X = X/255

##########################
#   ASSERTION CHECKING   #
##########################
assert(X.shape[0]==m)
assert(Y.shape[0]==m)


plt.figure()
plt.imshow(X[0])
plt.show()
plt.colorbar()
plt.grid(False)
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(56, 100)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# fucking sklearn won't split 3 ways, but i really like how clean the following is. to split 3 ways you provide 2 arguments, which must be the index number you want to split at, so out of m elements, you want to split at m*.8, or whatever
np.random.seed(0)
np.random.shuffle(X)
np.random.shuffle(Y)
train_X,dev_X,test_X = np.split(X, [int(.8*X.shape[0]), int(.9*X.shape[0])])
train_Y,dev_Y,test_Y = np.split(Y, [int(.8*X.shape[0]), int(.9*X.shape[0])])


# FIT THE MODEL
model.fit(train_X, train_Y, epochs=15)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


predictions = model.predict(test_images)
predictions.shape
np.argmax(predictions[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')




i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


img = test_images[0]

print(img.shape)

img = (np.expand_dims(img,0))

print(img.shape)


predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])
