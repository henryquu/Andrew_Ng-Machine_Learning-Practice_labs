import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math


def compute_cost(x_train, y_train, w, b):
    total_cost = np.sum((x_train * w + b - y_train)**2) / (2 * len(x_train))

    return total_cost

def test_cost(x_train, y_train):
    initial_w = 2
    initial_b = 1

    cost = compute_cost(x_train, y_train, initial_w, initial_b)

    if round(cost, 3) == 75.203:
        return True
    return False

def main():
    x_train, y_train = load_data()

    print(test_cost(x_train, y_train))
    

if __name__ == "__main__":
    main()
