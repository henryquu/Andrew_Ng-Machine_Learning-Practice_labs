from linear_regression import compute_cost, compute_gradient
from utils import *

def test_cost(x_train, y_train):
    initial_w = 2
    initial_b = 1

    cost = compute_cost(x_train, y_train, initial_w, initial_b)

    if round(cost, 3) == 75.203:
        print("Good cost.")
        return
    print("Wrong cost!")

def test_gradient(x_train, y_train):
    test_w = 0.2
    test_b = 0.2
    tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

    print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)

def main():
    x_train, y_train = load_data()

    test_cost(x_train, y_train)
    test_gradient(x_train, y_train)
    

if __name__ == "__main__":
    main()
