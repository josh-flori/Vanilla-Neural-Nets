# The purpose of this script is to create a tensorflow model on the same data as my hand-crafted network.
# Spoilers, I succeed.
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
X = np.array([imageio.imread(path+str(i)+'.jpg') for i in range(1,m+1)])
Y=np.array(pd.read_csv('/users/josh.flori/desktop/Y.csv')['Y'].values.tolist())
X.shape
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



##########################
#     VISUALIZE DATA     #
##########################
for i in range(10):
    print(Y[i])    
    plt.figure()
    plt.imshow(X[i])
    plt.show()


#####################
#     RANDOMIZE     #
#####################
np.random.seed(0)
np.random.shuffle(X)
np.random.seed(0)
np.random.shuffle(Y)



#######################
#      TRAIN TEST     #
#######################
train_X,dev_X,test_X = np.split(X, [int(.8*X.shape[0]), int(.9*X.shape[0])])
train_Y,dev_Y,test_Y = np.split(Y, [int(.8*X.shape[0]), int(.9*X.shape[0])])



########################
#     DEFINE MODEL     #
########################
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(35, 20,3)),
    keras.layers.Dense(50, activation=tf.nn.sigmoid),
    keras.layers.Dense(50, activation=tf.nn.sigmoid),
    keras.layers.Dense(50, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


########################
#     COMPILE MODEL    #
########################
model.compile(optimizer=tf.train.RMSPropOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

              
#####################
#      FIT MODEL    #
#####################
model.fit(train_X, train_Y, epochs=100,verbose=2,validation_data=(dev_X, dev_Y))


################################
#     EVALUATE ON TEST DATA    #
################################
model.evaluate(test_X, test_Y)



#########################################
#   PRODUCE THE PREDICTIONS, IF NEEDED  #
#########################################
predictions = model.predict(test_X)
predictions.shape
np.argmax(predictions[0])


