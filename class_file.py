import cv2
import glob
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class Dataset:
    def __init__(self, img_height, img_width, num_classes):
        # define the size and classes number
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        # create lists to restore img and labels
        self.X = []  # img
        self.y = []  # label

    def __len__(self):
        return len(self.X)

    def get_batch(self, start, end):
        return self.X[start:end], self.y[start:end]

    def loadData(self, address):
        # get img and label
        for folder in glob.glob(address):
            label = folder[-1]
            label = int(label)
            for img_path in glob.glob(folder + '/*.png'):
                img = plt.imread(img_path)
                img = cv2.resize(img, (self.img_height, self.img_width))
                self.X.append(img)
                self.y.append(label)
        # list to numpy
        self.X = np.array(self.X).reshape(100, -1)
        self.y = one_hot_encode(np.array(self.y), self.num_classes)


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type

    def forward(self, z):
        if self.activation_type == "sigmoid":
            return sigmoid(z)
        elif self.activation_type == "relu":
            return relu(z)
        else:
            raise ValueError("Unsupported activation function")

    def backward(self, dz, z):
        if self.activation_type == "sigmoid":
            return sigmoid_derivative(z)
        elif self.activation_type == "relu":
            return relu_derivative(z, dz)
        else:
            raise ValueError("Unsupported activation function")


class Neuron:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return z


class Layer:
    def __init__(self, input_size, output_size, activation_type="sigmoid"):
        self.n = Neuron(input_size, output_size)
        self.activation = Activation(activation_type)

    def forward(self, inputs):
        return self.activation.forward(self.n.forward(inputs))

    def backward(self, dz, inputs):
        dw = np.outer(inputs, dz)
        db = dz
        return dw, db


class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(input_size, hidden_size, activation_type="relu")
        self.output_layer = Layer(hidden_size, output_size, activation_type="sigmoid")

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        return self.output_layer.forward(hidden_output)

    def backward(self, predicted, actual, hidden_output, inputs):
        dz_output = predicted - actual
        dw_output, db_output = self.output_layer.backward(dz_output, hidden_output)

        dz_hidden = np.dot(dz_output, self.output_layer.n.weights.T) * hidden_output * (1 - hidden_output)
        dw_hidden, db_hidden = self.hidden_layer.backward(dz_hidden, inputs)

        return dw_hidden, db_hidden, dw_output, db_output


class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_weights(self, layer, gradient):
        layer.n.weights -= self.learning_rate * gradient


class Training:
    def __init__(self, model, ds, batch_size):
        self.model = model
        self.losses = []
        self.ds = ds
        self.batch_size = batch_size

    def train(self, epochs):
        for i in range(epochs):
            batch_x, batch_y = self.ds.get_batch(i, i + self.batch_size)
            predicted = softmax(self.model.forward(batch_x))
            loss = categorical_cross_entropy(batch_y, predicted)
            self.losses.append(loss)
            d_c = categorical_cross_entropy_derivative(batch_y, predicted)
            d_z = softmax_derivative(d_c)
            self.model.backward(d_z)


def read_in_batch(my_dataset, batch_size):
    for i in range(0, len(my_dataset), batch_size):
        batch = my_dataset.get_batch(i, i + batch_size)
        # 在这里进行数据处理，例如特征工程、模型训练等
        print("Batch size:", len(batch))
        print("First batch:", batch)


def normalize(x):
    return x / x.sum(axis=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x, dx):
    return dx * (x > 0)


def softmax(x):
    # Calculate activation x
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def softmax_derivative(s):



def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))


def categorical_cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true


# Define the one-hot encoding function
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def plot_loss_curve(training):
    # Plot the loss curve
    matplotlib.use('TkAgg')
    plt.plot(training.losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
