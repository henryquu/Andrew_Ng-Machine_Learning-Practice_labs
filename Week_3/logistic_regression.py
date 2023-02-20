import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math


def compute_cost(x, y, w, b, lamb=1):
    cases_count = len(y)

    func = sigmoid(np.dot(x, w) + b)

    cost = np.sum(-y * np.log(func) - (1 - y) * np.log(1 - func)) / cases_count
    
    return cost

def compute_gradient(x, y, w, b, lamb=1):
    cases_count = y.shape[0]
    feature_count = x.shape[1]

    func = sigmoid(np.dot(x, w) + b) - y
    b_integral = np.sum(func) / cases_count

    w_integrals = np.zeros(w.shape)
    for i in range(feature_count):
        w_integrals[i] = (np.sum(np.dot(func, x[:, i])) / cases_count)
    
    return b_integral, w_integrals

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    m = len(X)
    
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              

        if i<100000:
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history

def main():
    x_train, y_train = load_data("data/ex2data1.txt")

    n = x_train.shape[1]

    np.random.seed(1)
    intial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8

    iterations = 10000
    alpha = 0.001

    w,b, J_history,_ = gradient_descent(x_train ,y_train, intial_w, initial_b, 
                                    compute_cost, compute_gradient, alpha, iterations, 0)

if __name__ == '__main__':
    main()