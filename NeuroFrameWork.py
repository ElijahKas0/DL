import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder


class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)
        
         # Для Adam
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0  # Счетчик шагов
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation.forward(self.z) if self.activation else self.z
        return self.a
    
    def backward(self, grad_output, learning_rate, optimizer, momentum=0.9, clip_value=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        grad_activation = self.activation.backward(grad_output, self.z) if self.activation else grad_output
        grad_weights = np.dot(self.inputs.T, grad_activation)
        grad_biases = np.sum(grad_activation, axis=0, keepdims=True)
        grad_inputs = np.dot(grad_activation, self.weights.T)
        
        # print(f"Layer: {self}, Weight Grad Mean: {np.mean(grad_weights)}, Bias Grad Mean: {np.mean(grad_biases)}")
        
        # Gradient Clipping
        if optimizer == 'gradient_clipping':
            grad_weights = np.clip(grad_weights, -clip_value, clip_value)
            grad_biases = np.clip(grad_biases, -clip_value, clip_value)
        
        if optimizer == 'sgd':
            self.weights -= learning_rate * grad_weights
            self.biases -= learning_rate * grad_biases
        elif optimizer == 'momentum':
            self.velocity_w = momentum * self.velocity_w - learning_rate * grad_weights
            self.velocity_b = momentum * self.velocity_b - learning_rate * grad_biases
            self.weights += self.velocity_w
            self.biases += self.velocity_b
        elif optimizer == 'adam':
            self.t += 1
            self.m_w = beta1 * self.m_w + (1 - beta1) * grad_weights
            self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_weights ** 2)
            self.m_b = beta1 * self.m_b + (1 - beta1) * grad_biases
            self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_biases ** 2)

            # Bias correction
            m_w_hat = self.m_w / (1 - beta1 ** self.t)
            v_w_hat = self.v_w / (1 - beta2 ** self.t)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)

            self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        elif optimizer == 'gradient_clipping':
            self.weights -= learning_rate * grad_weights
            self.biases -= learning_rate * grad_biases
        
        return grad_inputs

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, grad_output, x):
        return grad_output * (x > 0)

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, grad_output, x):
        sigmoid_x = self.forward(x)
        return grad_output * sigmoid_x * (1 - sigmoid_x)

class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, grad_output, x):
        return grad_output  # Градиент вычисляется в функции потерь

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_grad, learning_rate, optimizer='sgd', momentum=0.9, clip_value=1.0):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate, optimizer, momentum, clip_value)
    
    def train(self, X, y, loss_fn, loss_fn_deriv, epochs=100, learning_rate=0.01, optimizer='sgd', momentum=0.9, batch_size=None, clip_value=1.0):
        for epoch in range(epochs):
            X, y = shuffle_data(X, y)
            if batch_size:
                batches = create_minibatches(X, y, batch_size)
            else:
                batches = [(X, y)]
            
            for X_batch, y_batch in batches:
                output = self.forward(X_batch)
                loss = loss_fn(y_batch, output)
                loss_grad = loss_fn_deriv(y_batch, output)
                self.backward(loss_grad, learning_rate, optimizer, momentum, clip_value)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Функции работы с данными
def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def create_minibatches(X, y, batch_size):
    batches = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        batches.append((X_batch, y_batch))
    return batches

def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Функции потерь
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

def cross_entropy_loss_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

# Загрузка датасета Iris
data = load_iris()
X, y = data.data, data.target

# One-hot encoding для меток классов
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

# Создание и настройка нейросети
model = NeuralNetwork()
model.add_layer(DenseLayer(4, 10, activation=ReLU()))
model.add_layer(DenseLayer(10, 3, activation=Softmax()))

# Обучение модели
model.train(X_train, y_train, loss_fn=cross_entropy_loss, loss_fn_deriv=cross_entropy_loss_derivative,
            epochs=170, learning_rate=0.01, optimizer='momentum', batch_size=16)

# Проверка на тестовых данных
predictions = model.forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Accuracy(точность) на Iris датасете: {accuracy * 100:.2f}%")




# Загрузка датасета MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Нормализация
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# One-hot encoding для меток классов
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Создание и настройка нейросети
model = NeuralNetwork()
model.add_layer(DenseLayer(784, 256, activation=ReLU()))
model.add_layer(DenseLayer(256, 128, activation=ReLU()))
model.add_layer(DenseLayer(128, 10, activation=Softmax()))

# Обучение модели
model.train(X_train, y_train, 
            loss_fn=cross_entropy_loss, 
            loss_fn_deriv=cross_entropy_loss_derivative,
            epochs=50, 
            learning_rate=0.01, 
            optimizer='momentum', 
            momentum=0.9, 
            batch_size=64)

# Проверка на тестовых данных
predictions = model.forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Accuracy(точность) на MNIST датасете: {accuracy * 100:.2f}%")
