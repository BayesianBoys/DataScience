import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

st.set_page_config(
    page_title="Gaussian Processes",
    page_icon="bayes_bois.png",
    initial_sidebar_state="expanded",
    )

st.image("bayes_bois.png", width = 100)

'''
# Gaussian Processes
'''

'''
Let's simulate data!
'''
expander = st.beta_expander("Trends")
with expander:
    lin_trend = st.slider("Linear trend", -50.0, 50.0, 0.4)
    poly_trend = st.slider("Polynomial Trend", -1.0, 1.0, 0.0)

expander = st.beta_expander("Seasonality")
with expander: 
    sinus = st.slider("Sinus", 0.0, 100.0, 1.0)
    sinus_2 = st.slider("Sinus 2", 0.0, 100.0, 1.0)
    sinus_2_period = st.slider("Sinus 2 Period", 0.0, 100.0, 0.4)

expander = st.beta_expander("Noise")

with expander:
    noise_amount = st.slider("Noise Amount", 1.0, 100.0, 1.0)

## FUNCTION 
def f(x):
    return lin_trend * x + sinus * np.sin(x) + sinus_2 * np.sin(sinus_2_period * x) + poly_trend * x**2

def get_RMSE(y, y_pred):
    listing = [(y[i]-y_pred[i])**2 for i in range(len(y))]
    return sum(listing)/len(listing)


def make_data(n):
    np.random.seed(42)

    x = np.linspace(0, 100, n)
    y = f(x)
    #y = [0.4 * x_i + 3 * np.sin(x_i) + 4 * np.sin(54 * x_i) + 5 * np.random.random(1)[0] for x_i in x]

    data = pd.DataFrame({"x": x,
                         "y": y})
    
    plotting = sns.lineplot(data = data, x = "x", y = "y")
    
    return data

from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct, RBF

kernel_select = st.sidebar.multiselect("Select Kernels", ["RationalQuadratic", "ExpSineSquared", "ExpSineSquared_2", "DotProduct", "WhiteKernel", "RBF", "Matern", "DotProduct Squared"], "RationalQuadratic")

### GLOBAL PARAMS

n = st.sidebar.number_input("Select number of points", 20)
x_end = st.sidebar.number_input("Select end for X", 10)
after_end = st.sidebar.number_input("Select end after X", 10)
test_n = st.sidebar.number_input("Select number of points in test", 20)

expander = st.beta_expander("Guide to exploring the app")

with expander:
    f'''
    ### Data Generation
    This app was made to showcase $sklearn$'s GaussianProcesses - an extremely useful method for time-series forecasting.
    Above you will find sliders, which specify the data generating function. Currently, the data-generating function is

    $$f(x) = {lin_trend} \cdot x + {poly_trend} \cdot x^2 + {sinus} \cdot sin(x) + {sinus_2} \cdot \sin({sinus_2_period} \cdot x)$$

    Optionally, the data generating function is not perfect. This can be used to simulate measurement errors. This is specified in
    the "Noise Amount"-slider. The data is sampled from a Gaussian distribution around the point ($x$), with a standard deviation equal to 
    the Noise Amount:

    $$x \sim N(f(x), {noise_amount})$$

    ### Kernels
    Kernels are the way Gaussian Processes are specified. As a starter intuition, these kernels can be seen as which distribution of functions,
    that the model is sampling from. The kernels can be combined by either multiplying or adding them together to create
    more complex kernels. In this app, all kernels are combined by adding them. For all kernels, $k_i$, you can specify their weight, $w_i$:

    $$k = \sum_i w_i \cdot k_i$$

    The only exception to this rule is the Dot Product Squared kernel. It is simply the Dot Product times itself.

    Some intuitions are useful here. Here is what we have learned so far. 

    The *Dot Product* can be viewed as a linear kernel. It can be used to sample from models that capture the linear trend in the data.

    The *Dot Product Squared* is maybe not surprisingly a polynomial kernel. It can be used to sample from models that capture a quadratic trend in the data.

    The *Exponential Sine Squared* kernels can be used to sample models that capture the periodic/cyclical relations in the data. The
    reason that there are two is to model two different cycles. The second Exponential Sine Squared Kernel can adjusted in terms of its periodicity.
    
    The *White Kernel* is white noise and can be used to model noise (i.e. measurement errors) in the data. Furthermore, it is useful to make the model fit the data.

    The *RGB* kernel treats points that are close to each other in $x$ as being close in $y$. 

    The *Rational Quadratic* is still a bit of a mystery. But it seems it might be useful if there was a non-linear trend that we needed to capture.

    Almost all kernels have an $l$-parameter, which can be understood as how local or global the kernel is. The larger $l$, the more global the kernel is. 

    ### Train and Test
    The model currently sees {n} points between 0 and {x_end}. The MSE is calculated for the test set. 

    ### Known Issues
    Currently, when parameters in the kernels are specified, for which there is no good solution, the kernel will default to the last value,
    which gave valid responses. If nothing you do changes the plot, please try to reset the kernels and start over. Another alternative fix is to 
    play around with the data by adding additional points or extending the range of the training data. 
    '''

## RATIONAL QUADRATIC

expander = st.sidebar.beta_expander("Rational Quadratic")
with expander:
    rational_weight = st.slider("Rational Quadratic Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
    rational = st.slider("Rational Quadratic Kernel L", min_value = 0.1, max_value = 100.0)
    alpha = st.slider("Rational Quadratic Alpha", min_value = 0.1, max_value = 100.0)

expander = st.sidebar.beta_expander("Dot Product")
with expander:
    dot_weight = st.slider("Dot Product Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
    sigma_dot = st.slider("Sigma for DotProduct", min_value = 0.0, max_value = 100.0)

expander = st.sidebar.beta_expander("Dot Product Squared")
with expander:
    dot_squared_weight = st.slider("Dot Product Squared Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
    squared_sigma = st.slider("Sigma for DotProduct Squared", min_value = 0.0, max_value = 100.0)

expander = st.sidebar.beta_expander("Exponential Sine Squared")
with expander:
    sine_weight = st.slider("Exp Sine Squared Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
    exp_sine_1 = st.slider("First Exp Sine Squared L", min_value = 0.1, max_value = 100.0)

expander = st.sidebar.beta_expander("Exponential Sine Squared 2")
with expander:
    sine_weight_2 = st.slider("Exp Sine Squared Weight 2", min_value = 0.1, max_value = 100.0, value = 1.0)
    exp_sine_2 = st.slider("Second Exp Sine Squared L", min_value = 0.1, max_value = 100.0)
    exp_sine_2_period = st.slider("Second Exp Sine Squared Period", min_value = 0.1, max_value = 100.0)

expander = st.sidebar.beta_expander("RBF")
with expander:
    rbf_weight = st.slider("RBF Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
    rbf = st.slider("RBF L", min_value = 0.1, max_value = 100.0)

expander = st.sidebar.beta_expander("Matern")
with expander:
    matern_weight = st.slider("Mattern Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
    matern_length = st.slider("Mattern Length", min_value = 0.1, max_value = 100.0, value = 1.0)
    matern_nu = st.slider("Mattern Nu", min_value = 0.1, max_value = 100.0, value = 1.5)

expander = st.sidebar.beta_expander("WhiteKernel")
with expander:
    white_weight = st.slider("White Noise Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
    white_noise = st.slider("NoiseLevel", min_value = 0.1, max_value = 10.0)


np.random.seed(1)

#def f(x):
 #   """The function to predict."""
  #  return x * np.sin(x)
if kernel_select:
    # ----------------------------------------------------------------------

    # Instantiate a Gaussian Process model

    interpret_dict = {
        "RationalQuadratic": rational_weight * RationalQuadratic(length_scale = rational, alpha = alpha),
        "ExpSineSquared": sine_weight * ExpSineSquared(length_scale = exp_sine_1),
        "ExpSineSquared_2": sine_weight_2 * ExpSineSquared(length_scale = exp_sine_2, periodicity= exp_sine_2_period),
        "DotProduct": dot_weight * DotProduct(sigma_0 = sigma_dot),
        "WhiteKernel": white_weight * WhiteKernel(white_noise),
        "RBF": rbf_weight * RBF(length_scale=rbf),
        "Matern": matern_weight * Matern(length_scale=matern_length, nu = matern_nu),
        "DotProduct Squared": dot_squared_weight * DotProduct(sigma_0=squared_sigma) * DotProduct(sigma_0=squared_sigma)
    }

    for i, ele in enumerate(kernel_select):
        element = interpret_dict.get(ele)
        if i == 0:
            kernel = element
        else:
            kernel += element

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # ----------------------------------------------------------------------
    # now the noisy case
    X = np.linspace(0.1, x_end, n)
    X = np.atleast_2d(X).T

    x = np.atleast_2d(np.linspace(0, x_end, 1000)).T

    #kernel = RationalQuadratic(length_scale = rational) + ExpSineSquared(length_scale = exp_sine_1) + ExpSineSquared(length_scale = exp_sine_2, periodicity= 4)

    # Observations and noise
    y = f(X).ravel()
    dy = 0.5 + noise_amount * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    X_test = np.linspace(x_end, x_end + after_end, test_n)
    X_test = np.atleast_2d(X_test).T

    y_test = f(X_test).ravel()
    dy_test = 0.5 + noise_amount * np.random.random(y_test.shape)
    noise = np.random.normal(0, dy_test)
    y_test += noise

    x_test = np.atleast_2d(np.linspace(x_end, x_end+after_end, 500)).T

    y_pred_test, sigma_test = gp.predict(x_test, return_std=True)

    y_predding, sigma_predding = gp.predict(X_test, return_std=True)

    errors = y_test - y_predding
    errors = errors.flatten()
    errors_mean = errors.mean()
    errors_std = errors.std()

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    fig, ax = plt.subplots(2, 1, figsize=(12, 10)) 
    ax[0].plot(x, f(x), 'r:', label=rf'$f(x) = {lin_trend} \cdot x + {poly_trend} \cdot x^2 +{sinus} \cdot \sin(x) + {sinus_2} \cdot \sin({sinus_2_period} \cdot x)$')
    ax[0].errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
    ax[0].plot(x, y_pred, 'b-', label='Prediction')
    ax[0].fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    ax[0].plot(x_test, f(x_test), 'r:', label='_nolegend_')
    ax[0].errorbar(X_test.ravel(), y_test, dy_test, fmt='r.', markersize=10, label='_nolegend_')
    ax[0].plot(x_test, y_pred_test, 'b-', label='_nolegend_')
    ax[0].fill(np.concatenate([x_test, x_test[::-1]]),
            np.concatenate([y_pred_test - 1.9600 * sigma_test,
                            (y_pred_test + 1.9600 * sigma_test)[::-1]]),
            alpha=.5, fc='b', ec='None', label='_nolegend_')
    #ax[0].xlabel('$x$')
    #ax[0].ylabel('$f(x)$')
    ax[0].axvline(x_end)
    ax[0].set(xlabel = "$x$", ylabel = "$f(x)$")
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          fancybox=True, shadow=True, ncol=2)
    ### HISTOGRAM:
    sns.distplot(a=errors, bins = 20, ax=ax[1])
    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--', label=f'$\mu$')
    ax[1].axvline(x=errors_mean + 2*errors_std, color=sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
    ax[1].axvline(x=errors_mean - 2*errors_std, color=sns_c[4], linestyle='--')
    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
    ax[1].legend()
    ax[1].set(title='Distribution of Error', xlabel='error', ylabel=None);
    st.pyplot(fig)

    col1, col2, col3 = st.beta_columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        stringing = str(gp.kernel)
        stringing = stringing.replace("**", "^")
        stringing = stringing.replace("*", "\cdot")
        stringing = stringing.replace("_", " ")

        #stringing = re.sub(r"**", "^", stringing)
        f'''Current settings: ${stringing}$'''
        f'''
        ## $$MSE = {get_RMSE(y_test, y_predding)}$$
        '''

    with col3:
        st.write("")
