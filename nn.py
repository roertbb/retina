import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pipeop import pipes

@pipes
def create_model(data):
    # data normalization
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Input is a 32×32 image with 3 color channels (red, green and blue)
    model = input_data(shape=data[0].shape, data_preprocessing=img_prep)
    # Step 1: Convolution
    model = conv_2d(model,32, 3, activation="relu")
    # Step 2: Max pooling
    model = max_pool_2d(model,2)
    # Step 3: Convolution again
    model = conv_2d(model,64, 3, activation="relu")
    # Step 4: Convolution yet again
    model = conv_2d(model,64, 3, activation="relu")
    # Step 5: Max pooling again
    model = max_pool_2d(model,2)
    # Step 6: Fully-connected 512 node neural network
    model = fully_connected(model,512, activation="relu")
    # Step 7: Dropout – throw away some data randomly during training to prevent over-fitting
    model = dropout(model,0.5)
    # Step 8: Fully-connected neural network with two outputs (model,0=isn’t a bird, 1=is a bird) to make the final prediction
    model = fully_connected(model, 2, activation="softmax")
    # Tell tflearn how we want to train the network
    model = regression(model,optimizer="adam", loss="categorical_crossentropy", learning_rate=0.001)
    # Wrap the network in a model object
    model = tflearn.DNN(model,tensorboard_verbose=0)

    return model

def train(model,X,Y,X_test,Y_test):
    Y = [[1,0] if y == 1 else [0,1] for y in Y]
    Y_test = [[1,0] if y == 1 else [0,1] for y in Y_test]
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(Y), n_epoch=1, shuffle=True, validation_set=(np.asarray(X_test, dtype=np.float64), np.asarray(Y_test)), show_metric=True, batch_size=32, snapshot_epoch=True)
    return model

def predict(model, img, sample_size = 32):
    h, w, _ = img.shape
    img = cv2.resize(img,(int(h/8),int(w/8)))
    h, w, _ = img.shape
    padding = int(sample_size/2)
    
    print(w,h)
    pr = np.zeros((h,w))
    for i in range(padding, h-padding):
        for j in range(padding, w-padding):
            model.predict(np.asarray([img[i-padding:i+padding, j-padding:j+padding]], dtype=np.float64))
        
    plt.imshow(pr)
    plt.show()
    
    
def load(model, file):
    return model.load(file)

def save(model, file):
    model.save(file)
    return model





