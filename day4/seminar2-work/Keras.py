import numpy as np
from mnist import load_dataset
import keras
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
y_train,y_val,y_test = map(keras.utils.np_utils.to_categorical,[y_train,y_val,y_test])


import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


import keras
from keras.models import Sequential
import keras.layers as ll

model = Sequential(name="cnn")

model.add(ll.InputLayer([1,28,28]))

model.add(ll.Reshape([28,28,1]))
model.add(ll.Conv2D(32, (5,5),))
model.add(ll.Activation('relu'))
model.add(ll.MaxPool2D((2,2),))

model.add(ll.Flatten())

#network body
model.add(ll.Dense(25))
model.add(ll.Activation('linear'))

model.add(ll.Dropout(0.9))

model.add(ll.Dense(25))
model.add(ll.Activation('linear'))

#output layer: 10 neurons for each class with softmax
model.add(ll.Dense(10,activation='softmax'))

model.compile("adam","categorical_crossentropy",metrics=["accuracy"])

model.summary()

#fit(X,y) with a neat automatic logging. Highly customizable under the hood.
model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=5)

model.save("weights.h5")

print("\nLoss, Accuracy = ",model.evaluate(X_test,y_test))
