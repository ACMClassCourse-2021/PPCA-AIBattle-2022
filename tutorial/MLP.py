import numpy as np
from matplotlib import pyplot as plt

"""
    STEP 1: Define the base class of the neural network layer.
"""

# Base class of the nnet layer
class Layer():
    def __init__(self):
        pass
    
    # Forward propagation function: compute the output by input x
    def forward(self, x):
        raise NotImplementedError
    
    # Backward propagation function: compute dE/dW and dE/dx by node_grad(dE/dy)
    def backward(self, node_grad):
        raise NotImplementedError
    
    # Update function: update the weights by gradients
    def update(self, learning_rate):
        raise NotImplementedError

"""
    STEP 2: Implement the activation functions.
"""

class Sigmoid(Layer):    
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    
    def backward(self, node_grad):
        return node_grad * (self.y * (1 - self.y))
    
    def update(self, learning_rate):
        pass

class Relu():        
    def forward(self, x):
        """
            TODO: Finish the forward propagation function and save some variables if you need
        """

    
    def backward(self, node_grad):
        """
            TODO: Finish the backward propagation function 
        """
    
    def update(self, learning_rate):
        pass

class Softmax_Cross_Entropy():
    def forward(self, x):
        """
            TODO: Finish the forward propagation function and save some variables if you need
        """
    
    def backward(self, label):
        """
            TODO: Finish the backward propagation function 
        """
    
    def update(self, learning_rate):
        pass

"""
    STEP 3: Implement the linear layer.
"""

class Linear(Layer):    
    def __init__(self, size_in, size_out, with_bias):
        self.size_in = size_in
        self.size_out = size_out
        self.with_bias = with_bias
        self.W = self.initialize_weight()
        if with_bias:
            self.b = np.zeros(size_out)
    
    
    def initialize_weight(self):
        epsilon = np.sqrt(2.0 / (self.size_in + self.size_out))
        return epsilon * (np.random.rand(self.size_in, self.size_out) * 2 - 1)
        
    def forward(self, x):
        """
            TODO: Finish the forward propagation function and save some variables if you need
        """
    
    def backward(self, node_grad):
        """
            TODO: Finish the backward propagation function and save gradients of W and b
        """
    
    def update(self, learning_rate):
        """
            TODO: Update W and b by gradients calculated in the backward propagation function
        """

"""
    STEP 4: Combine all parts into the MLP.
"""

class MLP():    
    def __init__(self, layer_size, with_bias=True, activation="sigmoid", learning_rate=1):
        assert len(layer_size) >= 2
        self.layer_size = layer_size
        self.with_bias = with_bias
        if activation == "sigmoid":
            self.activation = Sigmoid
        elif activation == "relu":
            self.activation = Relu
        else:
            raise Exception("activation not implemented")
        self.learning_rate = learning_rate
        self.build_model()
        
    def build_model(self):
        self.layers = []
        
        size_in = self.layer_size[0]
        for hu in self.layer_size[1:-1]:
            self.layers.append(Linear(size_in, hu, self.with_bias))
            self.layers.append(self.activation())
            size_in = hu
            
        self.layers.append(Linear(size_in, self.layer_size[-1], self.with_bias))
        self.layers.append(Softmax_Cross_Entropy())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, label):
        node_grad = label
        for layer in reversed(self.layers):
            node_grad = layer.backward(node_grad)
            
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
            
    def train(self, x, label):
        y = self.forward(x)
        self.backward(label)
        self.update(self.learning_rate)
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x)
    
    def loss(self, x, label):
        y = self.forward(x)
        return -np.log(y) @ label


"""
    STEP 5: Test
"""

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])

np.random.seed(1007)
EPOCH = 1000
N = X.shape[0]

mlp = MLP([2, 4, 2], learning_rate=0.1, activation="relu")

loss = np.zeros(EPOCH)
for epoch in range(EPOCH):
    for i in range(N):
        mlp.train(X[i], Y[i])
        
    for i in range(N):
        loss[epoch] += mlp.loss(X[i], Y[i])
        
    loss[epoch] /= N
    
plt.figure()
ix = np.arange(EPOCH)
plt.plot(ix, loss)
plt.show()