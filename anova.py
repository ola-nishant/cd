import pandas as pd
from scipy.stats import f_oneway

# Assuming you have the Iris dataset available in sklearn
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# One-way ANOVA test
def one_way_anova(feature, target):
    groups = [iris_df[feature][iris_df['target'] == i] for i in range(3)]
    f_statistic, p_value = f_oneway(*groups)
    return f_statistic, p_value

# Example usage
feature1 = 'sepal length (cm)'
target = 'target'
f_statistic_one_way, p_value_one_way = one_way_anova(feature1, target)
print(f"One-way ANOVA: F-statistic = {f_statistic_one_way}, p-value = {p_value_one_way}")
