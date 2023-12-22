import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
target = iris.target

# Function to generate data from a binomial distribution and plot its PDF
def generate_and_plot_binomial(n, p):
    data = np.random.binomial(n, p, size=1000)
    unique, counts = np.unique(data, return_counts=True)
    pdf = counts / len(data)

    plt.bar(unique, pdf, label=f'n={n}, p={p}', alpha=0.5)
    plt.title('Binomial Distribution PDF')
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

# Example usage
generate_and_plot_binomial(n=10, p=0.5)
generate_and_plot_binomial(n=20, p=0.3)
generate_and_plot_binomial(n=15, p=0.7)
