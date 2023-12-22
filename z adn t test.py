import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# Plot the data distribution
sns.pairplot(sns.load_dataset('iris'), hue='species')
plt.show()

# Function to perform t-test
def t_test(sample1, sample2):
    _, p_value = stats.ttest_ind(sample1, sample2)
    return p_value

# Extracting samples for two different classes (e.g., setosa and versicolor)
setosa_sample = data[target == 0][:, 0]  # Taking only the first feature for demonstration
versicolor_sample = data[target == 1][:, 0]

# Perform t-test
t_test_p_value = t_test(setosa_sample, versicolor_sample)
print(f'T-Test p-value: {t_test_p_value}')

# Interpret the results
alpha = 0.05
if t_test_p_value < alpha:
    print("Reject null hypothesis (significant difference) based on T-Test")
else:
    print("Fail to reject null hypothesis (no significant difference) based on T-Test")
