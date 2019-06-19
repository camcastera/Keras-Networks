# Based on LeNet by Y. Le Cun : http://yann.lecun.com/exdb/lenet/


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

def create_model(load=False, save=False, file='LeNet.h5', print=False):
    model = Sequential()

    model.add(Conv2D(filters = 6, 
                     kernel_size = 5, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = (32,32,3)))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Conv2D(filters = 16, 
                     kernel_size = 5,
                     strides = 1,
                     activation = 'relu',
                     input_shape = (14,14,6)))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    #Flatten
    model.add(Flatten())
    model.add(Dense(units = 120, activation = 'relu'))
    model.add(Dense(units = 84, activation = 'relu'))

    #Output Layer
    model.add(Dense(units = 10, activation = 'softmax'))

    # Compile the model with the SGD optimizer:

    model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if print:
        model.summary()
        
    # Save or Load Initial weights 
    if save:
        model.save_weights(file)
    if load:
        model.load_weights(file)
    return model
