import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

## FUNCTION 
def f(x):
    return 0.4 * x + 3 * np.sin(x) + 4 * np.sin(3 * x)

def make_data(n):
    np.random.seed(42)

    x = np.linspace(0, 100, n)
    y = f(x)
    #y = [0.4 * x_i + 3 * np.sin(x_i) + 4 * np.sin(54 * x_i) + 5 * np.random.random(1)[0] for x_i in x]

    data = pd.DataFrame({"x": x,
                         "y": y})
    
    sns.lineplot(data = data, x = "x", y = "y")
    
    return data

from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct

kernel_select = st.sidebar.multiselect("Select Kernels", ["RationalQuadratic", "ExpSineSquared", "ExpSineSquared(period = 4)", "DotProduct", "WhiteKernel"], "RationalQuadratic")
n = st.sidebar.number_input("Select number of points", 20)
x_end = st.sidebar.number_input("Select end for X", 10)
after_end = st.sidebar.number_input("Select end after X", 10)
rational = st.sidebar.slider("Rational Quadratic Kernel L", min_value = 0.1, max_value = 100.0)
alpha = st.sidebar.slider("Rational Quadratic Alpha", min_value = 0.1, max_value = 100.0)
sigma_dot = st.sidebar.slider("Sigma for DotProduct", min_value = 0.0, max_value = 100.0)
exp_sine_1 = st.sidebar.slider("First Exp Sine Squared L", min_value = 0.1, max_value = 100.0)
exp_sine_2 = st.sidebar.slider("Second Exp Sine Squared L", min_value = 0.1, max_value = 100.0)

np.random.seed(1)

#def f(x):
 #   """The function to predict."""
  #  return x * np.sin(x)

# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate a Gaussian Process model

interpret_dict = {
    "RationalQuadratic": RationalQuadratic(length_scale = rational, alpha = alpha),
    "ExpSineSquared": ExpSineSquared(length_scale = exp_sine_1),
    "ExpSineSquared(period = 4)": ExpSineSquared(length_scale = exp_sine_2, periodicity= 4),
    "DotProduct": DotProduct(sigma_0 = sigma_dot),
    "WhiteKernel": WhiteKernel()
}

for i, ele in enumerate(kernel_select):
    element = interpret_dict.get(ele)
    if i == 0:
        kernel = element
    else:
        kernel += element

#kernel = RationalQuadratic(length_scale = rational, alpha = alpha) + ExpSineSquared(length_scale = exp_sine_1) + ExpSineSquared(length_scale = exp_sine_2, periodicity= 4)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = 0.4 \cdot x + 3 \cdot sin(x) + 4 * \sin(3 \cdot x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

# ----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, x_end, n)
X = np.atleast_2d(X).T

x = np.atleast_2d(np.linspace(0, x_end, 1000)).T

#kernel = RationalQuadratic(length_scale = rational) + ExpSineSquared(length_scale = exp_sine_1) + ExpSineSquared(length_scale = exp_sine_2, periodicity= 4)

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                              n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

X_test = np.linspace(x_end, x_end + after_end, 50)
X_test = np.atleast_2d(X_test).T

y_test = f(X_test).ravel()
dy_test = 0.5 + 1.0 * np.random.random(y_test.shape)
noise = np.random.normal(0, dy_test)
y_test += noise

x_test = np.atleast_2d(np.linspace(x_end, x_end+after_end, 500)).T

y_pred_test, sigma_test = gp.predict(x_test, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = 0.4 \cdot x + 3 \cdot sin(x) + 4 * \sin(3 \cdot x)$')
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.plot(x_test, f(x_test), 'r:', label='_nolegend_')
plt.errorbar(X_test.ravel(), y_test, dy_test, fmt='r.', markersize=10, label='_nolegend_')
plt.plot(x_test, y_pred_test, 'b-', label='_nolegend_')
plt.fill(np.concatenate([x_test, x_test[::-1]]),
         np.concatenate([y_pred_test - 1.9600 * sigma_test,
                        (y_pred_test + 1.9600 * sigma_test)[::-1]]),
         alpha=.5, fc='b', ec='None', label='_nolegend_')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.axvline(x_end)
plt.legend(loc='upper left')
st.pyplot(plt)