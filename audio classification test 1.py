# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import numpy as np
import keras
import librosa
import scipy.io.wavfile as wav
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Load the sound data
wav_file = 'SparkTest1.wav'
sample_rate, data = wav.read(wav_file)

# Extract the features from the sound data
mfccs = librosa.feature.mfcc(data, sr=sample_rate)

# Save the features to a .npy file
np.save('feat.npy', mfccs)

# Load the sound data
X = np.load('feat.npy')
y = np.load('label.npy').ravel()

# Get the number of classes
num_classes = np.max(y, axis=0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Convert label to onehot
y_train = keras.utils.to_categorical(y_train-1, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test-1, num_classes=num_classes)

# Build the Neural Network
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=193))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=64)

# Evaluate the model
score, acc = model.evaluate(X_test, y_test, batch_size=32)

# Print the test score and accuracy
print('Test score:', score)
print('Test accuracy:', acc)