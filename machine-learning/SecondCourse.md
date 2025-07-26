## Training a Logistic Regression
Step 1: Specify how to compute output given input x and parameters w and b
Step 2: Specify loss and cost (Eg: Logistic?)
Step 3: Train on data to minimise Cost Function with grad desc

# Training a Neual Network
Step 1: model = Sequential(Dense(units=.activation=), Dense())
Step 2: model.compile(loss=)
Step 3: model.fit(X, Y, epochs=) >> Automatically does back propagation

J(W, B) = Cost of every parameter in the neural network

ReLU Activation Function: if z<0 g(z)=0, if z>=0 g(z)=z, g(z)=max(0,z). If the model is predicting a non-negative values. Most common hidden layer activation function. 
>> When the gradient is flat in more places gradient descent is flat. Since ReLU only has 1 flat portion it will lead to faster learning.

Linear Activation Function: g(z)=z, like no activation function. For Regression problems that predict a positive or negative value meaning there is a direction. If used, the model will just become a linear regression. Dont use in hidden layers.
Eg:
a1 = wx+b
a2 = w(a1)+b = wx+b (Still Linear!!!)

Softmax Regression:
a1 = e^z1/(e^z1 + e^z2 + e^z3 + e^z4 +...) where z1=w1.x+b1 for all a values where a is the probability of P(y=1) where y can be all possible values for prediction, sum_of_all(a)=1. Softmax regression makes sure all probabilities adds up to 1.

Loss for Softmax Regression (Sparse Categorical Cross Entropy Loss):
loss = {-loga1 if y = 1, -loga2 if y = 2, ..., -logaN if y = N}
Derivation: loss = -yloga1 - (1-y)log(1-a1)
                     |              |
                if y = 1    if y=0, 1-a1=a2
Softmax Output Layer: Softmax is applied to last layer to return a probability distribution. The softmax depends on the value of all z.

Numerical Roundoff Errors: 
Dense(units=1, activation="linear")
model.compile(loss=BinaryCrossEntropy(from_logits=True))
>> Tensorflow internally applies sigmoid, softmax etc in a numerically stable way and thus all we need is to output raw logits. Best practice in binary classification logits.

Multi-class vs Multi-label Classification?

Adaptive Moment Estimation (Adam): Automatically adjusts learning rate as you train for each parameter separately. There is still a need to set an initial learning rate. Eg:
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=)

Dense Layers: Every neuron is connected to every neuron in the next layer and use the previous layer's activations as input
Convolutional Layers: Used in image and signal processing tasks, each neuron looks at only a small region of the input (receptive field) rather than the whole input. As each neuron has fewer inputs, computation is faster. They help neural networks requrie less training data and be less prone to overfitting which means less memorisation. 

d/dw(J(w))=k, if w increases by 0.001, J(w) = loss is times k.

J(w,b) = 1/2(a-y)^2
If N nodes and P parameters, compute derivates in roughly N+P steps rather than NxP steps using computational graph.

