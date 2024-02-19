# Gradient-Descent
I'm try to write a gradient descent algorithm with python for Simple Linear Regression

# What is Gradient Descent
Gradient Descent is an optimization algorithm, is widely used in machine learning to find parameter values <sub>(for linear regression parameters are coefficent w and intercept b)</sub> that minimize the cost function. 
here is steps of gradient descend algorithm:
1. Compute the derivative of cost function for each parameter.
2. Assign step size<sub> num_iter</sub> and learning rate<sub> alpha</sub> values.
3. Assign random values to the parameters.
4. Apply the gradient descent algorithm `w = w - alpha * dj_dw` for coefficent w and `b = b - alpha * dj_db` for intertercept b.
5. Repeat in the amount of step size.
