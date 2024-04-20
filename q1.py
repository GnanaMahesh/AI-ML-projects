import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    N,a,b = X.shape
    X1 = X.reshape(N,a*b)
    mean = np.mean(X1, axis=0)
    # Center the dataset
    centered_X1 = X1 - mean
     # Compute the covariance matrix
    covariance_matrix = np.cov(centered_X1, rowvar=False)
   # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
   # Sort eigenvectors based on eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # Select the top k eigenvectors
    return sorted_eigenvectors[:, :k]
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    # Project data points onto the subspace formed by the principal components
    N,a,b = X.shape
    X1 = X.reshape(N,a*b)
    projected_X = np.dot(X1,basis)
    return projected_X
    # END TODO


if __name__ == '__main__':
    mnist_data = 'mnist.pkl'
    with open(mnist_data, 'rb') as f:
        data = pkl.load(f)
    # Now you can work with the loaded object
    X=data[0]
    y=data[1]
    k=10
    basis = pca(X,k)
    print(projection(X,basis))
    