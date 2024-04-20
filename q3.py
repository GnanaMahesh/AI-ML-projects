import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def initialise(self, X_train):
        """
        Initialize the self.centroids class variable, using the "k-means++" method, 
        Pick a random data point as the first centroid,
        Pick the next centroids with probability directly proportional to their distance from the closest centroid
        Function returns self.centroids as an np.array
        USE np.random for any random number generation that you may require 
        (Generate no more than K random numbers). 
        Do NOT use the random module at ALL!
        """
        # TODO
        self.centroids = [X_train[np.random.randint(X_train.shape[0])]]
        while len(self.centroids) < self.n_clusters:
            distances = np.array([min([np.linalg.norm(point - centroid)**2 for centroid in self.centroids]) for point in X_train])
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for idx, cumulative_probability in enumerate(cumulative_probabilities):
                if r < cumulative_probability:
                    self.centroids.append(X_train[idx])
                    break
        self.centroids = np.array(self.centroids)
        return self.centroids
        # END TODO
    def fit(self, X_train):
        """
        Updates the self.centroids class variable using the two-step iterative algorithm on the X_train dataset.
        X_train has dimensions (N,d) where N is the number of samples and each point belongs to d dimensions
        Ensure that the total number of iterations does not exceed self.max_iter
        Function returns self.centroids as an np array
        """
        # TODO
        for _ in range(self.max_iter):
            classifications = []
            for point in X_train:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                classification = np.argmin(distances)
                classifications.append(classification)
            prev_centroids = np.copy(self.centroids)
            for i in range(self.n_clusters):
                points = [X_train[j] for j in range(len(X_train)) if classifications[j] == i]
                if len(points) > 0:
                    self.centroids[i] = np.mean(points, axis=0)
            # Check for convergence
            if np.sum(np.abs(prev_centroids - self.centroids)) < 1e-6:
                break
        return self.centroids

        # END TODO
    
    def evaluate(self, X):
        """
        Given N data samples in X, find the cluster that each point belongs to 
        using the self.centroids class variable as the centroids.
        Return two np arrays, the first being self.centroids 
        and the second is an array having length equal to the number of data points 
        and each entry being between 0 and K-1 (both inclusive) where K is number of clusters.
        """
        # TODO
        classifications = []
        for point in X:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            classification = np.argmin(distances)
            classifications.append(classification)
        return self.centroids, np.array(classifications)
        # END TODO

def evaluate_loss(X, centroids, classification):
    loss = 0
    for idx, point in enumerate(X):
        loss += np.linalg.norm(point - centroids[classification[idx]])
    return loss

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed+1)

    random_state = random.randint(10,1000)
    centers = 5

    X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=seed)
    X_train = StandardScaler().fit_transform(X_train)

    # Fit centroids to dataset
    kmeans = KMeans(n_clusters=centers)
    kmeans.initialise(X_train)
    kmeans.fit(X_train)
    print(kmeans.evaluate(X_train))
    class_centers, classification = kmeans.evaluate(X_train)
    
    #print(evaluate_loss(X_train,class_centers,classification))

    # View results
    sns.scatterplot(x=[X[0] for X in X_train],
                    y=[X[1] for X in X_train],
                    hue=true_labels,
                    style=classification,
                    palette="deep",
                    legend=None
                    )
    plt.plot([x for x, _ in kmeans.centroids],
            [y for _, y in kmeans.centroids],
            'k+',
            markersize=10,
            )
    plt.savefig("hello.png")