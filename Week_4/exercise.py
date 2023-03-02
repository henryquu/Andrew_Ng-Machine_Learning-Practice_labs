import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def my_dense(a_in, W, b, g):
    z = a_in @ W + b

    return g(z)

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)

if __name__ == "__main__":
    X, y = load_data()

    training_count, features_count = X.shape

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network implemented in Numpy
        my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
        my_yhat = int(my_prediction >= 0.5)

        # Predict using the Neural Network implemented in Tensorflow
        tf_prediction = model.predict(X[random_index].reshape(1,400))
        tf_yhat = int(tf_prediction >= 0.5)
            
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{tf_yhat},{my_yhat}")
        ax.set_axis_off() 
    fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
    plt.show()