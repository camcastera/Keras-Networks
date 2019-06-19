import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Based on Network in Network : https://openreview.net/forum?id=ylE6yojDR5yqX
# Implemented by Camille Castera: 
#          Repository: http://github.com/camcastera/ , Personnal Page:  https://camcastera.github.io/

# if you use this, please cite the authors paper, and the github repository.



def create_model(load=False, save=False, file='NiN.h5', nbclass=10, dimshape=(32,32,3), theano=False, print=False, loss='sparse_categorical_crossentropy'):
    if theano:
        order = "channels_first"
        dimshape = (dimshape[2],dimshape[0],dimshape[1])
    else:
        order = "channels_last"
    
    model = Sequential()
    
    #First CNN with MaxPooling
    model.add(Conv2D(filters = 192, 
                     kernel_size = 5, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = dimshape,
                     )
              )
    model.add(Conv2D(filters = 160, 
                     kernel_size = 1, 
                     strides = 1,
                     activation = 'relu',
                      input_shape = dimshape,
                     )
              )
    model.add(Conv2D(filters = 96, 
                     kernel_size = 1, 
                     strides = 1, 
                     activation = 'relu',
                     input_shape = dimshape,
                     )
                )
    model.add(MaxPooling2D(pool_size = 3, strides = 2))
    model.add(Dropout(0.5))
    
    # Second CNN with AVG Pooling
    model.add(Conv2D(filters = 192, 
                     kernel_size = 5, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = dimshape,
                     )
              )
    model.add(Conv2D(filters = 192, 
                     kernel_size = 1, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = dimshape,
                     )
              )
    model.add(Conv2D(filters = 192, 
                     kernel_size = 1, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = dimshape,
                     )
                )
    model.add(AveragePooling2D(pool_size = 3, strides = 2))
    model.add(Dropout(0.5))
    
    #Third and last CNN, with AVG Pooling
    model.add(Conv2D(filters = 192, 
                     kernel_size = 3, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = dimshape,
                     )
              )
    model.add(Conv2D(filters = 192, 
                     kernel_size = 1, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = dimshape,
                     )
              )
    
    model.add(Conv2D(filters = nbclass, 
                     kernel_size = 1, 
                     strides = 1, 
                     activation = 'relu',
                     input_shape = dimshape,
                     )
                )
    model.add(AveragePooling2D(pool_size = 2, strides = 1))
    
    # Flatten
    model.add(Flatten())
    
    #This layer is not in the original paper
    model.add(Dense(units = nbclass, activation = 'softmax'))
    
    '''
    # Add a non-trainable Dense linear layer to prevent a Keras-Tensorflow issue
    
    model.add(Dense(units = nbclass, activation = 'linear',trainable=False))
    # Set all the weights to 1, so this is the identity layer
    
    w = model.layers[-1].get_weights()
    for i in range(w[0].shape[0]):
        for j in range(w[0].shape[1]):
            w[0][i,j]=1.0
    model.layers[-1].set_weights(w)
    '''
    
    #Compile
    model.compile(loss=loss,optimizer='SGD',metrics=['accuracy'])
    
    # You can print the summary of the model if necessary
    if print:
        model.summary()
        
    # Save or Load Initial weights 
    if save:
        model.save_weights(file)
    if load:
        model.load_weights(file)
    return model


#Usage : 

#model = create_model()

