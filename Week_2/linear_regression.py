import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math


def compute_cost(x_train, y_train, w, b):
    total_cost = np.sum((x_train * w + b - y_train) ** 2) / (2 * x_train.shape[0])

    return total_cost


def compute_gradient(x_train, y_train, w, b):
    m = x_train.shape[0]

    dj_dw = np.sum((x_train * w + b - y_train) * x_train) / m
    dj_db = np.sum(x_train * w + b - y_train) / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)  

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history


def main():
    x_train, y_train = load_data()
    initial_w = 0.
    initial_b = 0.

    iterations = 1500   
    alpha = 0.01

    w, b, _, _ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                        compute_cost, compute_gradient, alpha, iterations)
    print("w,b found by gradient descent:", w, b)
    
    m = x_train.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w * x_train[i] + b

    # Plot the linear fit
    plt.plot(x_train, predicted, c = "b")

    # Create a scatter plot of the data. 
    plt.scatter(x_train, y_train, marker='x', c='r') 

    # Set the title
    plt.title("Profits vs. Population per city")
    # Set the y-axis label
    plt.ylabel('Profit in $10,000')
    # Set the x-axis label
    plt.xlabel('Population of City in 10,000s')
    plt.show()

    predict1 = 3.5 * w + b
    print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

    predict2 = 7.0 * w + b
    print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

if __name__ == "__main__":
    main()
