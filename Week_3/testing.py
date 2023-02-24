from logistic_regression import *
from utils import *
from public_tests import *
from reg_log_regression import *

def main():
    sigmoid_test(sigmoid)
    compute_cost_test(compute_cost)
    compute_gradient_test(compute_gradient)
    predict_test(predict)

    compute_cost_reg_test(compute_cost_reg)
    compute_gradient_reg_test(compute_gradient_reg)

if __name__ == '__main__':
    main()
