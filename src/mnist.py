import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_val, y_val = X_train[40000:], y_train[40000:]
X_train, y_train = X_train[:40000], y_train[:40000]
print(X_train.shape)

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

Y_train = to_categorical(y_train, 10)
Y_val = to_categorical(y_val, 10)
Y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'sigmoid', input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), activation = 'sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
H= model.fit(X_train, Y_train, validation_data = (X_val, Y_val), batch_size=32, epochs=10,verbose=1)
score = model.evaluate(X_test, Y_test, verbose = 0)
print(score)
plt.imshow(X_test[0])
y_predict= model.predict(X_test[0].reshape(1,32,32,3))
print('gia tri du doan: ', np.argmax(y_predict))