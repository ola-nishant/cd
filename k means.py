import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-means clustering algorithm.
    Parameters:
    - X: Input data, a 2D array where each row represents a data point.
    - k: Number of clusters.
    - max_iters: Maximum number of iterations.
    - tol: Tolerance to declare convergence.
    Returns:
    - centroids: Final cluster centroids.
    - labels: Cluster assignments for each data point.
    """
    # Initialize centroids randomly
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return centroids, labels


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Selecting only the first two features for simplicity
y = iris.target

# Apply K-means clustering
k = 3
centroids, labels = kmeans(X, k)

# Visualize the results
plt.figure(figsize=(12, 6))

# Plot the original data with true labels
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=70)
plt.title('Original Data with True Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the clustered data with K-means labels
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', s=70)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
