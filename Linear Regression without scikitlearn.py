import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_regression(x,w,b):
    m = x.shape[0]
    x_pred = []
    for i in range(m):
        f_wb = b + x[i] * w
        x_pred.append(f_wb)
    return x_pred

def cost_function(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = b + x[i] * w
        cost = cost + (f_wb - y[i])**2
    total_cost = 1/(2 * m) * cost
    return total_cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    # i cant find a way to get derivative functions by automatic because there is a for loop in cost function
    for i in range(m):
        f_wb = b + x[i] * w
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

def gradient_function(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x,y,w,b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])

    return w,b,J_history,p_history

x_train = np.array([1.0,1.2,1.4,2.5])
y_train = np.array([500.0,237.0,350.5,465.7])
w_init = 0
b_init = 0
num_iter = 10000
alpha = 0.001
w,b,J_history,p_history = gradient_function(x_train,y_train,w_init,b_init,alpha,num_iter,cost_function,gradient_function)


cost_init = cost_function(x_train,y_train,w_init,b_init)
cost_gradient = cost_function(x_train,y_train,w,b)

print(cost_init)
print(cost_gradient)
