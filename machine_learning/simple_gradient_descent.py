import numpy as np
import matplotlib.pyplot as plt

# Simple gradient descent for linear regression

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        prediction = w * x[i] + b
        error = prediction - y[i]
        cost += error ** 2
    cost = cost / (2 * m)
    return cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        prediction = w * x[i] + b           # Current prediction
        error = prediction - y[i]           # Error
        
        dj_dw += error * x[i]               # Gradient for w
        dj_db += error                      # Gradient for b
    
    dj_dw = dj_dw / m  # Average the gradients
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, learning_rate, iterations):
    w = w_init
    b = b_init
    costs = []
    
    for i in range(iterations): # Epoch
        cost = compute_cost(x, y, w, b)
        costs.append(cost)
        
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # Update parameters (take a step downhill)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        # Print progress every 1000 iterations
        if i % 1000 == 0:
            print(f"Iteration {i}: Cost = {cost:.2f}, w = {w:.2f}, b = {b:.2f}")
    
    return w, b, costs

# Run gradient descent
print("Starting gradient descent...")
w_final, b_final, cost_history = gradient_descent(
    x_train, y_train, 
    w_init=0, b_init=0, 
    learning_rate=0.01, 
    iterations=10000
)

print(f"\nFinal results:")
print(f"w (slope) = {w_final:.2f}")
print(f"b (intercept) = {b_final:.2f}")
print(f"Final cost = {cost_history[-1]:.2f}")

# Test our model
print(f"\nPredictions:")
print(f"1000 sqft house: ${w_final * 1.0 + b_final:.0f}k")
print(f"1200 sqft house: ${w_final * 1.2 + b_final:.0f}k") 
print(f"2000 sqft house: ${w_final * 2.0 + b_final:.0f}k")
