import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.math import confusion_matrix
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
from PIL import Image
(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()

X_train.shape

y_train

#scaling the values between 0 & 1
X_train=X_train/255
X_test=X_test/255

#creating model and adding layers
model = Sequential()
model.add(Input(shape=(28,28)))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='softmax'))

#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

#training the model
model.fit(X_train,y_train,epochs=15,validation_split=0.2)

loss,accuracy=model.evaluate(X_test,y_test)
print(accuracy)

#testing
plt.imshow(X_test[4])
plt.show()

#calculating the probabilities for all labels
y_prob=model.predict(X_test)

#perdicting the correct label
y_pred=np.argmax(y_prob,axis=1)
print(y_pred[707])

#real time prediction
path='5.png'
input_image=cv2.imread(path)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
grayscale=cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
img_resized=cv2.resize(grayscale,(28,28),interpolation=cv2.INTER_NEAREST)
img_resized=img_resized/255
img_resized = 1 - img_resized
img_reshaped=np.reshape(img_resized,[1,28,28])
input_pred=model.predict(img_reshaped)
input_pred_label=np.argmax(input_pred,axis=1)
print('the handwritten label is ',input_pred_label)