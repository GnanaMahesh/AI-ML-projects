import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # # TODO
        n, d, _ = X.shape
        X = X.reshape(n, -1)
        classes = np.unique(y)
        num_classes = len(classes)
        class_means = np.array([np.mean(X[y == c], axis=0) for c in classes])
        overall_mean = np.mean(X, axis=0)
        Sw = np.zeros((d*d, d*d))
        for c in classes:
            class_data = X[y == c]
            class_mean = class_means[c]
            scatter_matrix = np.dot((class_data - class_mean).T, (class_data - class_mean))
            Sw += scatter_matrix

        # Compute between-class scatter matrix Sb
        Sb = np.zeros((d*d, d*d))
        for c in classes:
            n_c = X[y == c].shape[0]
            class_mean = class_means[c].reshape(-1, 1)
            overall_mean_reshaped = overall_mean.reshape(-1, 1) 
            Sb += n_c * np.outer((class_mean - overall_mean_reshaped), (class_mean - overall_mean_reshaped))
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
        eig_pairs = []
        for i in range(len(eigenvalues)):
            eig_pairs.append((np.abs(eigenvalues[i]), eigenvectors[:,i]))
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        linear_discriminants = []
        for i in range(0, self.n_components):
            linear_discriminants.append(eig_pairs[i][1].reshape(d*d, 1))
        self.linear_discriminants = np.hstack(linear_discriminants)
        return self.linear_discriminants  
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        projected = np.dot(X.reshape(len(X), -1), w)
        return projected                   # Modify as required
        # END TODO

if __name__ == '__main__':
    mnist_data = 'mnist.pkl'
    with open(mnist_data, 'rb') as f:
        data = pkl.load(f)
    X=data[0]
    y=data[1]
    k=10
    lda = LDA(k)
    w=lda.fit(X, y)
    X_lda = lda.transform(X,w)
