import numpy as np
from sklearn.cluster import KMeans

def load_data(path):
    with open(path, 'r') as file:
        file.readline()

        parameters = file.readline().split()

        P, N, M = map(lambda x: int(x.split('=')[1]), parameters[1:])

        data = np.loadtxt(file, dtype=np.dtype(np.float64))
    #data = np.subtract(data, 0.1) #for some reason, 0.1 is added to all values in the file?
    # Extract input and output matrices
    input_matrix = data[:, :N]
    output_matrix = data[:, N:]

    return input_matrix, output_matrix

class RBFN:

    def __init__(self, hidden_shape, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(center, data_point)
        return G

    def _choose_centers(self, X):
        kmeans = KMeans(n_clusters=self.hidden_shape)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

    def fit(self, X, Y):
        self._choose_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

if __name__ == "__main__":
    # Load training data
    train_input, train_output = load_data('training_data.txt')

    # Initialize RBFN
    rbfn = RBFN(hidden_shape=10, sigma=1.0)

    # Train RBFN
    rbfn.fit(train_input, train_output)

    # Load test data
    test_input, test_output = load_data('test_data.txt')

    # Test RBFN
    predictions = rbfn.predict(test_input)

    # Print predictions
    print(predictions)
