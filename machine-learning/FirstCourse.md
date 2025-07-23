Automatic Convergence Test: Let epsilon, e, be a value. If loss decreases by less than or equals to e in 1 iteration, declare convergence. Meaning you have found a global minimum
>> If learning rate is too large, loss will increase with iteration
>> New Weight = Old Weight - Learning Rate * rate of change of loss wrt to Weight
>> New Bias = Old Bias - Learning Rate * rate of change of loss wrt to Bias

Creating New Features: Use intuition to esign new features for easier learning

Polynomial Regression: 
Polynomial Equation: f(x) = wx + wx^2 + wx^3 + b

def compute_gradient(x, y, w, b): 
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        error = f_wb_i - y[i]
        dj_dw += error * x[i] >> Tweaks the direction of the slope based on x-value
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

Logistic Regression
Sigmoid/Logistic Function: Outputs y values between 0 and 1. f(x) = g(w.x+b).
f(x)=P(y=1) or P(y=0)+P(y=1)=1: probabilities add up to 1. 

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

Decision Boundaries and Non-linear Decision Boundaries
>> Linear Regression Cost (Convex): Mean Squared Error to converge to minima
>> Logistic Regression Cost (Non-Convex): Cost = -[actual×log(predicted) + (1-actual)×log(1-predicted)]
>> When y = 1 (Yes): L(f(x), y) = -log(f(x))
>> When y = 0 (No): L(f(x), y) = -log(1-f(x))
>> Intuition: When y = 1 (Yes) predicted and true label is 0 (No) loss becomes almost infintie vice versa for both cases of the logistic loss function. Look at graph shape

Logistic Regression Loss Function:
L = -y*log(prediction) - (1-y)*log(1-prediction) 
-1/m * sum_of_1_to_m(L = -y*log(prediction) - (1-y)*log(1-prediction))

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost

def compute_gradient_logistic(X, y, w, b): 
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)
        err_i  = f_wb_i  - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j] >> How much each feature contributes to error (part of the gradient of logistic loss curve: ∂Loss/∂wⱼ = (fᵢ − yᵢ) * xᵢⱼ)
        dj_db = dj_db + err_i >> Since b doesnt multiply any feature its gradient is just the raw error (∂Loss/∂b = (fᵢ − yᵢ))
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_db, dj_dw  

Gradient Descent for Logistic Regression: (1/m) sum_of_all(f(x)-y) or (1/m) sum_of_all(f(x)-y)x

High Bias = Underfit
Overfit = High Variance (Not too high polynomials!)
Too many features + insufficient features = Overfit

Regularisation: Lets you keep the features and reduces the weights of the features to small values. Small weights->Simpler Model->Better Generalisation
>> Adds λ/2m * sum_of_1_to_n(w^2) or even λ/2m * sum_of_1_to_n(b^2) to loss function
>> Controlled by the magnitude of λ!!! The higher to stronger the regularisation
Regularisation Gradient Descent: loss = loss + λ/m * w (for dJ_dw) is not applicable to b, bias. 
>> There is no regularisation for bias, b!!!

### Linear Regression with Regularisation
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b=scalar, see np.dot
        cost = cost + (f_wb_i - y[i])**2            
    cost = cost / (2 * m) 
    reg_cost = 0    >>> + λ/2m * sum_of_1_to_n(w^2)
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
    total_cost = cost + reg_cost
    return total_cost


### Logistic Regression with Regularisation
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b=scalar, see np.dot
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)  
    cost = cost/m
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
    total_cost = cost + reg_cost
    return total_cost

# Gradient Descent for Regularised Linear Regression (Same for logistic regression)
def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]                 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]               
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m   
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]   >>> + λ/m * w for every dj_dw
        # Or can use numpy's vectorized in-built: dj_dw += (lambda_ / m) * w
    return dj_db, dj_dw
