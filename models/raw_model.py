import numpy as np

# =====================
# LINEAR REGRESSION
# =====================

def linear_regression_train(X, y, learning_rate=0.01, epochs=100):
    # Initialize weight and bias
    weight = 0.0
    bias = 0.0
    m = len(X)  # number of samples
    for _ in range(epochs):
        # Model prediction: y = weight * x + bias
        y_pred = weight * X + bias
        # Compute gradients (derivatives of cost wrt weight and bias)
        d_weight = (2/m) * np.sum((y_pred - y) * X)
        d_bias = (2/m) * np.sum(y_pred - y)
        # Update parameters
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias
    return weight, bias

def linear_regression_predict(X, weight, bias):
    return weight * X + bias

# =====================
# LOGISTIC REGRESSION (as a single neuron)
# =====================

def sigmoid(z):
    # Sigmoid activation squashes input to [0, 1]
    return 1 / (1 + np.exp(-z))

def logistic_regression_train(X, y, learning_rate=0.1, epochs=100):
    # Initialize weight and bias for the neuron
    weight = 0.0
    bias = 0.0
    m = len(X)
    for _ in range(epochs):
        # Neuron output: z = weight * x + bias
        z = weight * X + bias
        # Activation: a = sigmoid(z)
        activation = sigmoid(z)
        # Compute gradients (derivatives of cost wrt weight and bias)
        d_weight = (1/m) * np.sum((activation - y) * X)
        d_bias = (1/m) * np.sum(activation - y)
        # Update parameters
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias
    return weight, bias

def logistic_regression_predict(X, weight, bias):
    # Output probability (between 0 and 1)
    z = weight * X + bias
    return sigmoid(z)

# =====================
# EXAMPLE USAGE (uncomment to run)
# =====================
# X = np.array([1, 2, 3, 4, 5])
# y_linear = np.array([2, 4, 6, 8, 10])
# weight, bias = linear_regression_train(X, y_linear)
# print('Linear Regression Prediction:', linear_regression_predict(X, weight, bias))
#
# y_logistic = np.array([0, 0, 0, 1, 1])
# weight, bias = logistic_regression_train(X, y_logistic)
# print('Logistic Regression Probability:', logistic_regression_predict(X, weight, bias))
# print('Logistic Regression Class:', logistic_regression_predict(X, weight, bias) > 0.5)

# =====================
# DENSE NEURAL NETWORK (Pure Python, no numpy, no OOP)
# =====================

def sigmoid(z):
    # Sigmoid activation for a single value
    return 1 / (1 + pow(2.718281828459045, -z))

def relu(z):
    # ReLU activation for a single value
    return z if z > 0 else 0

def dense_layer(inputs, weights, biases, activation):
    # inputs: list of floats (input vector)
    # weights: list of list of floats (shape: num_inputs x num_neurons)
    # biases: list of floats (one per neuron)
    # activation: function to apply (sigmoid or relu)
    outputs = []
    for neuron in range(len(biases)):
        z = 0
        for i in range(len(inputs)):
            z += inputs[i] * weights[i][neuron]
        z += biases[neuron]
        outputs.append(activation(z))
    return outputs

# Example: 2-layer NN (input -> hidden -> output)
def simple_dense_nn_forward(x):
    # Example weights and biases for demonstration
    # Input layer: 2 features
    # Hidden layer: 3 neurons
    # Output layer: 1 neuron
    W1 = [[0.1, 0.2, 0.3],  # weights from input 1 to each hidden neuron
          [0.4, 0.5, 0.6]]  # weights from input 2 to each hidden neuron
    b1 = [0.1, 0.2, 0.3]    # biases for hidden layer
    W2 = [[0.7], [0.8], [0.9]]  # weights from each hidden neuron to output
    b2 = [0.4]                  # bias for output neuron
    # Forward pass
    hidden_activations = dense_layer(x, W1, b1, relu)
    output_activations = dense_layer(hidden_activations, W2, b2, sigmoid)
    return output_activations

# Example usage (uncomment to run):
# x = [1.0, 2.0]  # input vector
# print('NN output:', simple_dense_nn_forward(x))

# =====================
# GRADIENT COMPUTATION (for a single neuron, pure Python)
# =====================
def compute_single_neuron_gradient(x, y_true, w, b):
    # Forward pass
    z = 0
    for i in range(len(x)):
        z += x[i] * w[i]
    z += b
    a = sigmoid(z)
    # Binary cross-entropy loss: L = -[y*log(a) + (1-y)*log(1-a)]
    # Compute gradients
    dL_da = -(y_true / a) + ((1 - y_true) / (1 - a))
    da_dz = a * (1 - a)  # derivative of sigmoid
    dL_dz = dL_da * da_dz
    dL_dw = [dL_dz * x[i] for i in range(len(x))]
    dL_db = dL_dz
    return dL_dw, dL_db

# Example usage (uncomment to run):
# x = [1.0, 2.0]
# y_true = 1
# w = [0.1, 0.2]
# b = 0.3
# grad_w, grad_b = compute_single_neuron_gradient(x, y_true, w, b)
# print('Gradient w.r.t weights:', grad_w)
# print('Gradient w.r.t bias:', grad_b)
