#Libraries importing

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Dataset loading
data = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# Flattening data to reduce size of dataset
f_x_train = x_train.reshape(len(x_train), 28 * 28).astype('float32') / 255
f_x_test = x_test.reshape(len(x_test), 28 * 28).astype('float32') / 255

# Model creation
model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling model for controling the loss and optimizing the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(f_x_train, y_train,epochs=5,batch_size=32,validation_split=0.2)

#Evaluating 
model.evaluate(f_x_test, y_test, verbose=2)

#Prediction
y_prediction = model.predict(f_x_test)

y_prediction_lab = [np.argmax(i) for i in y_prediction]

#Confusion matrix
conMat= tf.math.confusion_matrix(y_test, y_prediction_lab, num_classes=10)

# Plotting 
plt.figure(figsize=(10, 8))
sns.heatmap(conMat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Actual value')
plt.ylabel('Predicted Value')