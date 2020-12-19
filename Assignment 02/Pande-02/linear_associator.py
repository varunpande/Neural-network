# Pande, Varunkumar
# 1001-722-538
# 2020-10-11
# Assignment-02-01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        if transfer_function.lower() == "hard_limit" or transfer_function.lower() == "linear":
            self.transfer_function = transfer_function.lower()
        else:
            raise Exception("Invalid Transfer Function. Please Enter either Hard_limit or Linear")

        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
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
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        summation = np.dot(self.weights,X)

        if self.transfer_function == "hard_limit":
            # applying activation function - hardlimit
            summation = np.where(summation < 0, 0, 1)
            return summation
        elif self.transfer_function == "linear":
            return summation
    
    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        MP_psuedoinverse = np.linalg.pinv(X)
        self.weights = np.dot(y,MP_psuedoinverse)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        if learning.lower() == 'filtered':
            new_weight = lambda gamma,old_weight,alpha,t_q,a_q,P_q : np.add((1 - gamma)*old_weight,alpha*np.dot(t_q,P_q.T))
        elif learning.lower() == 'delta':
            new_weight = lambda gamma,old_weight,alpha,t_q,a_q,P_q : np.add(old_weight,alpha*np.dot(t_q - a_q,P_q.T))
        elif learning.lower() == 'unsupervised_hebb':
            new_weight = lambda gamma,old_weight,alpha,t_q,a_q,P_q : np.add(old_weight,alpha*np.dot(a_q,P_q.T))
        else:
            raise Exception('Please select a valid learning rule Possible methods are Filtered,Delta,Unsupervised_hebb')

        # split the batch of input
        split_input=np.array_split(X,batch_size,axis=1)

        for _ in range(num_epochs):
            # split the batch of target output
            split_target = np.array_split(y,batch_size,axis=1)

            for i in range(X.shape[1]//batch_size):
                actual = self.predict(split_input[i])
                new_weights = new_weight(gamma,self.weights,alpha,split_target[i],actual,split_input[i])
                self.set_weights(new_weights)

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        actual_output = self.predict(X)
        mean_squared_error = np.mean((y - actual_output) ** 2)
        return mean_squared_error
