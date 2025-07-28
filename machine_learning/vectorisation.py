
# Dot Product
# Dot product of x.y is the same as the vector multiplication
# of transpose(x) @ y sums the result into a single number
# Transpose is matrix.T

# Vector Matrix Multiplication

# Matrix @ Matrix Multiplication
# Matrix multiplication combines rows of the first matrix with columns of the 
# second by multiplying corresponding elements and summing 
# them to produce each element in the result.

# The number of rows of W should be the dimensions of the input activation

def dense(AT, W, b):
    z = np.matmul(AT, W) + b
    a_out = g(z)
    return a_out

# without vectorization
for j in range(0, 16):
    f = f + w[j] * x[j] # Sequential and slow

# with vectorization
np.dot(w,x) # Retrieves all elements in both vectors and executes in parallel
w = w-0.1*d # Auto parrallel process and auto assign to w.
