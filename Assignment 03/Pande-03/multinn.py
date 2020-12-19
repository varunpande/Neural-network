# Pande, Varunkumar
# 1001-722-538
# 2020-10-25
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        # list of weights for all the layers
        self.weights = []
        # list of biases for all layers
        self.biases = []
        # list of activation functions for all layers
        self.activations = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.weights.append(tf.Variable(np.random.randn(self.input_dimension, num_nodes)))
        self.biases.append(tf.Variable(np.random.randn(num_nodes)))
        self.input_dimension=num_nodes

        #  check for allowed activation fucntions 
        allowed_transfer_func = {"linear","relu","sigmoid"}
        if transfer_function.lower() in allowed_transfer_func:
            self.activations.append(transfer_function.lower())
        else: 
            raise Exception('No such activation function found: ', transfer_function)

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        copy_of_weights = tf.identity(self.weights[layer_number])
        return copy_of_weights.numpy()

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        copy_of_biases = tf.identity(self.biases[layer_number])
        return copy_of_biases.numpy()

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        if (self.weights[layer_number].shape[0] == weights.shape[0]) and (self.weights[layer_number].shape[1] == weights.shape[1]):
            self.weights[layer_number] = tf.Variable(weights)

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        if (self.biases[layer_number].shape[0] == biases.shape[0]) and (len(biases.shape) == 1):
            self.biases[layer_number] = tf.Variable(biases)

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))#).numpy()#)

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        input_to_layer = X
        for i in range(len(self.weights)):
            net = tf.matmul(input_to_layer, self.weights[i]) + self.biases[i]
            if self.activations[i] == 'sigmoid':
                output_a = tf.nn.sigmoid(net)
            elif self.activations[i] == 'linear':
                output_a = net
            elif self.activations[i] == 'relu':
                output_a = tf.nn.relu(net)
            input_to_layer = output_a
        output_of_multilayer = input_to_layer
        return output_of_multilayer

    @tf.function
    def train_on_batch(self, x, y, alpha):
        with tf.GradientTape(persistent=True) as tape:
            predictions_of_layer = self.predict(x)
            loss = self.calculate_loss(y, predictions_of_layer)
            for layer_number in range(len(self.weights)):
                dloss_dw, dloss_db = tape.gradient(loss, [self.weights[layer_number], self.biases[layer_number]])
                self.weights[layer_number].assign_sub(alpha * dloss_dw)
                self.biases[layer_number].assign_sub(alpha * dloss_db)

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        tf_batched_input=tf.data.Dataset.from_tensor_slices((X_train, y_train))
        tf_batched_input=tf_batched_input.batch(batch_size)
        for epoch in range(num_epochs):
            for _,(x,y) in enumerate(tf_batched_input):
                self.train_on_batch(x, y, alpha)

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        number_of_errors = 0
        predictions_of_layer = self.predict(X)
        predictions_of_layer = predictions_of_layer.numpy()
        for i in range(predictions_of_layer.shape[0]):
            if np.argmax(predictions_of_layer[i]) != y[i]:
                number_of_errors += 1
        percent_error = number_of_errors/X.shape[0]
        return percent_error

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        predictions_of_layer = self.predict(X)
        # initializing a confusion matrix of dimension [number_of_classes,number_of_classes]
        confusion_matrix = np.zeros((predictions_of_layer.shape[1],predictions_of_layer.shape[1]))
        for col, row in enumerate(y):
            index_of_max_output = np.argmax(predictions_of_layer[col])
            confusion_matrix[row][index_of_max_output] += 1
        return confusion_matrix