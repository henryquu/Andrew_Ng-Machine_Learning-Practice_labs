import numpy as np
import matplotlib.pyplot as plt
from utils import *
from logistic_regression import compute_cost, compute_gradient, gradient_descent, predict
import math

def compute_cost_reg(X, y, w, b, lambda_ = 1):
    cases_count = X.shape[0]

    without_reg = compute_cost(X, y, w, b, lambda_)
    regularization = lambda_ * np.sum(w**2) / (2 * cases_count)

    return without_reg + regularization

def compute_gradient_reg(X, y, w, b, lambda_ = 1):
    cases_count = X.shape[0]

    b_integral, w_no_reg = compute_gradient(X, y, w, b, lambda_)

    reg = lambda_ * w / cases_count

    return b_integral, w_no_reg + reg

def main():
    X_train, y_train = load_data("data/ex2data2.txt")

    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])

    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1])-0.5
    initial_b = 1.


    lambda_ = 0.01;                                          
    iterations = 10000
    alpha = 0.01

    w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                        compute_cost_reg, compute_gradient_reg, 
                                        alpha, iterations, lambda_)

    plot_decision_boundary(w, b, X_mapped, y_train)
    plt.show()

    p = predict(X_mapped, w, b)

    print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

if __name__ == '__main__':
    main()