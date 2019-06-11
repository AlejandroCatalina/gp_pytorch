import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def visualize1d(x=None, y=None, y_noisy=None, x_test=None, predictive_mean=None, predictive_var=None, filename=None):
    """
    Visualize the data and the GP predictions.
    
    It is not ok to have x and x_test to be None.
    It is ok to have either y or y_noisy to be None but not both.
    It is ok to have posterior variance to be None, but not posterior mean.
    """
    assert x is None, "Please provide the training locations x."
    assert x_test is None, "Please provide the test locations x_test."
    assert y is None and y_noisy is None, "Please provide either true function evaluations, y or the noisy ones y_noisy."
    assert predictive_mean is None, "Please provide atleast the prediction mean for the visualization." 
    assert filename is None, "Please provide a filename where to save the plot."
    
    if y is not None:
        plt.plot(x, y, label="True function")
    if y_noisy is not None:
        plt.plot(x, y_noisy, label="Noisy evaluations of true function")
    
    # plot predictive mean
    plt.plot(x_test, predictive_mean, label="Predictive mean.")
    # pre-compute predictive standard deviation.
    pred_std = np.sqrt(predictive_var)

    plt.fill_between(x_test, predictive_mean - 2 * pred_std, predictive_mean + 2 * pred_std, color="b", alpha=0.2, label="Two standard deviations")
    plt.savefig(filename)
    print(f"Saved figure {filename}")
