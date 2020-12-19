# Pande, Varunkumar
# 1001-722-538
# 2020-09-27
# Assignment-01-01

import numpy as np

class SingleLayerNN(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.initialize_weights()

    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions + 1)

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        if (W.shape[0] == self.weights.shape[0]) and (W.shape[1] == self.weights.shape[1]):
            self.weights = np.copy(W)
        else:
            return -1

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        copy_of_weight_matrix = np.copy(self.weights)
        return copy_of_weight_matrix

    def predict(self, X):
        """
        Make a prediction on a batch of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        # creating a bias array to prepend into the input
        bias_array = np.ones((1,X.shape[1]))
        X = np.append(bias_array,X,axis=0)
        summation = np.dot(self.weights,X)
        # applying activation function - hardlimit
        summation = np.where(summation < 0, 0, 1)
        return summation

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        # checking for exceptions
        if (X.shape[1] != Y.shape[1]) and (Y.shape[0] != self.weights.shape[0]):
            raise Exception('input and output dimensions mismatch')
        elif (alpha < 0.0 or alpha > 1.0):
            raise Exception('alpha should be in the range of 0.0 to 1.0')
        elif (num_epochs <= 0):
            raise Exception('Number of Epochs should be greater than 0')
        X_copy = np.copy(X)
        # creating a bias array to prepend into the input
        bias_array = np.ones((1,X_copy.shape[1]))
        X_copy = np.append(bias_array,X_copy,axis=0)
        for i in range(num_epochs):
            target = self.predict(X)
            error_matrix =  Y - target
            w_adjustment = alpha * np.dot(error_matrix,X_copy.T)
            new_weights = np.add(self.weights,w_adjustment)
            self.set_weights(new_weights)

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
        target = self.predict(X) 
        error_matrix = np.sum(np.logical_xor(Y,target), axis=0)
        error_matrix = np.array(error_matrix, dtype = bool)
        number_of_errors = sum(error_matrix)
        error_percent = 100*(number_of_errors/Y.shape[1])
        return error_percent


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())