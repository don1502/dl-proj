#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

#Importing and splitting of datasets
model_ = keras.models.Model
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
x_train=x_train.reshape(len(x_train), 28*28)
x_test=x_test.reshape(len(x_test), 28*28)

#function for visualizing the results
def show_visual ( data, title, n=10, h=28, w=28):
  plt.figure(figsize=(10,5))
  for i in range(n):
    plt.subplot(1,n,i+1)
    plt.imshow(data[i].reshape(h,w), cmap='gray')

#Defining dimensions of the model
input_dim, output_dim = 784, 784
encoding_dim = 100
hidden_dim = 256

#Creating the model layers
input_layer = keras.layers.Input(shape=(input_dim,), name='input')
hidden_layer_1 = keras.layers.Dense(hidden_dim, activation='relu', name='Hidden_1')(input_layer)
bottle_neck = keras.layers.Dense(encoding_dim, activation='relu', name='Bottle_Neck')(hidden_layer_1)
hidden_layer_2 = keras.layers.Dense(hidden_dim, activation='relu', name='Hidden_2')(bottle_neck)
output_layer = keras.layers.Dense(output_dim, activation='sigmoid', name='Output')(hidden_layer_2)

model= model_(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.summary()      if u want to see the summary of the model use this line

#Training the model
model.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))
decoded_data = model.predict(x_test)
get_encoded_data = keras.models.Model(inputs=model.input, outputs=model.get_layer('Bottle_Neck').output)
encoded_data = get_encoded_data.predict(x_test)

#calling the function to visualize the results
show_visual(x_test, 'original')
show_visual(encoded_data, 'encoded_data', h=10, w=10)
show_visual(decoded_data, 'decoded_data')

