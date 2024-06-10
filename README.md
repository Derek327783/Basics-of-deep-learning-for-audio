# Deep Learning for Audio
- This repository contains notes and code for implementing Multi-layer Perceptron, Convolutional Neural Networks and Recurrent Neural Networks as a Music Genre Classifier.

## 1. Dataset
- The dataset contains 30 second snippets of audio files from different genres ranging from rock to jazz.

## 2. Feature Extraction
- The pipeline first splits the 30 second audio files into 5 segments. Afterwards MFCCs are extracted from each of the 5 segments. Each of the segments are labelled the same genre as the audio file they were taken from.

## 3. Optimizing neural networks
- **Normal Gradient Descent**: Updating of weight only occurs after the enire training set gets fed forward. The final gradient that is used is the average of all the derivatives attained after passing through each training sample.
- **Batch Gradient Descent**: Updating of weights is done after every batch.
- **Stochastic Gradient Descent**: Updating of weights is done after every sample.
 
## 4. Multi-layer Perceptron
- Below is the architecture of the MLP.
```
MLP = keras.Sequential()

MLP.add(keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
MLP.add(keras.layers.Dense(512, activation="relu"))
MLP.add(keras.layers.Dense(256, activation="relu"))
MLP.add(keras.layers.Dense(64, activation="relu" ))
MLP.add(keras.layers.Dense(10, activation="softmax"))

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
MLP.compile(optimizer="adam", loss = "sparse_categorical_crossentropy",metrics=["accuracy"])
```
- Dimension of x_input = (130,13) n_frames * n_mfccs
- Flatten = 130 x 13 = 1690, (1x1690)
- Dimension of W_1 = (1690x512) dim_x * n_neurons
- Dimension of W_2 = (512x256) 
- Dimension of W_3 = (256x64)
- Dimension of last_layer = 10

## 5. Convolutional Neural Network
- Below is the architecture of the CNN.
```
CNN = keras.Sequential()
CNN.add(keras.layers.Conv2D(32, (3,3), activation="relu",input_shape=(X_train.shape[1],X_train.shape[2],1)))
CNN.add(keras.layers.MaxPooling2D((3,3), strides=(2,2),padding="same"))
CNN.add(keras.layers.Conv2D(64,(3,3),activation ="relu"))
CNN.add(keras.layers.MaxPooling2D((3,3), strides=(2,2),padding="same"))
CNN.add(keras.layers.Conv2D(128,(2,2),activation ="relu"))
CNN.add(keras.layers.Flatten())
CNN.add(keras.layers.Dense(64,activation="relu"))
CNN.add(keras.layers.Dropout(0.1))
CNN.add(keras.layers.Dense(10,activation="softmax"))
``` 
- **Dimension of x_input**: (130x13)
- **Formula to calculate dimension of output layer of Convolutional or MaxPooling layer**: ((Input Dimension - kernel Dimension + 2 * Padding)/Stride)+1
- An error was ran into because the dimension of the output of the layer is smaller than the kernel which is why the last layer is (2,2) instead of (3,3)

## 6. Recurrent Neural Networks/LSTMs
- Below is the architecture of the LSTM
```
LSTM = keras.Sequential()
LSTM.add(keras.layers.LSTM(64, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
LSTM.add(keras.layers.LSTM(64))
LSTM.add(keras.layers.Dense(64, activation='relu'))
LSTM.add(keras.layers.Dropout(0.3))
LSTM.add(keras.layers.Dense(10, activation='softmax'))
```
- The main thing that allowed me to understand this was that the memory cell that is used in a RNN is the same memomry cell at each time step so it is similar to SGD in the sense that each time step we are looking at one sample.
- Another thing that helped med understand it was that unlike the previous 2 neural network architectures you cant connect everything through neural network relations i.e. matrix multiplication. Specifically the equation ht = f(xWxh + ht-1Whh+bt). View the x and as like two separate neural networks.

