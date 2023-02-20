from logistic_regression import *
from utils import *
from public_tests import *

def main():
    sigmoid_test(sigmoid)
    compute_cost_test(compute_cost)
    compute_gradient_test(compute_gradient)


if __name__ == '__main__':
    main()
