import numpy as np
import matplotlib.pyplot as plt

# 1. Building the Neural Network Model
# 2. Forward Propagation

def dense(a_in, W, b):
    units = W.shape[1] # Number of output units
    a_out = np.zeros(units) # Pre-allocate output vector
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

# If network has 𝑠𝑖𝑛 units in a layer and  𝑠𝑜𝑢𝑡 units in the next layer, then
# 𝑊 will be of dimension  𝑠𝑖𝑛×𝑠𝑜𝑢𝑡
# 𝑏 will a vector with  𝑠𝑜𝑢𝑡 elements

def sequential(x):
    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a1, W3, b3)
    fx = a3
    return fx

model = sequential(input_data)
