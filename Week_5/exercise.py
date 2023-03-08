import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer

import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from public_tests import * 

from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)

def my_softmax(z):
    ez = np.exp(z)
    return ez / np.sum(ez)

def main():
    X, y = load_data()
    
    tf.random.set_seed(1234)

    model = Sequential([
        InputLayer((400, )),
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=10, activation='linear')
        ], name='model1'
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam'
    )

    test_model(model, 10, 400)

    model.fit(X, y, epochs=40)

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, _ = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
    widgvis(fig)
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)
        
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
        ax.set_axis_off()
    
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

    print( f"{display_errors(model,X,y)} errors out of {len(X)} images")

if __name__ == "__main__":
    main()
