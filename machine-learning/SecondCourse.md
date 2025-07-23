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



