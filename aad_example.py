import numpy as np

#def f(x, y):
#    w1 = x**2
#    w2 = 2*x*y
#    return w1 + w2

# Define the function we want to approximate: f(x, y) = x^2 + 2xy
def f(x, y):
    return x**2 + 2*x*y

def aad_example(x, y):
    # Forward pass
    w1 = x**2
    w2 = 2*x*y
    f_value = w1 + w2

    # Backward pass (Adjoint Algorithmic Differentiation)
    dw1 = 1
    dw2 = 1

    dx = dw1 * (2*x) + dw2 * (2*y)
    dy = dw2 * (2*x)

    return dx, dy


'''
# Example usage
x = 2
y = 3

# Compute function value
result = f(x, y)
print("f(x, y) =", result)

# Compute derivatives using AAD
dx, dy = aad_example(x, y)
print("df/dx =", dx)
print("df/dy =", dy)
'''


# Generate some random training data
np.random.seed(42)
x_train = np.random.uniform(low=-1, high=1, size=(100, 2))
y_train = f(x_train[:, 0], x_train[:, 1])


# Define the neural network model
class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(2)
        print('weights: ', self.weights)

    def forward(self, x):
        return np.dot(x, self.weights)

    def backward(self, x, y):
        # Compute derivatives using AAD
        dx, dy = aad_example(x[0], x[1])
        dw = np.array([dx, dy])

        # Update model parameters using gradients
        self.weights -= 0.01 * (self.forward(x) - y) * dw

    def train(self, x_train, y_train, epochs):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                self.backward(x, y)

# Create an instance of the neural network
model = NeuralNetwork()

# Train the model
model.train(x_train, y_train, epochs=100)

# Test the model
x_test = np.array([[1, 2], [-1, 3], [0, 0]])
y_test = f(x_test[:, 0], x_test[:, 1])
predictions = model.forward(x_test)

# Print the test results
for x, y, pred in zip(x_test, y_test, predictions):
    print(f"Input: {x}, Target: {y}, Prediction: {pred}")
