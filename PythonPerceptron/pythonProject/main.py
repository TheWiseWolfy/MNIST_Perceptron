import numpy as np
import resource

from keras.datasets import mnist
import time

class Perceptron:
    def __init__(self, input_size, output_size, activation_function, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function

        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros(output_size)
        print(f"Memory used in perceptron: {max_rss_megabytes - baseline:.2f} MB")

    def predict(self, inputs):

        summation = np.dot(inputs, self.weights) + self.bias
        output = self.activation_function(summation)

        return output
    def train(self, training_inputs, training_outputs):
        start_time = time.time()

        for epoch in range(self.epochs):
            for training_input, training_output in zip(training_inputs, training_outputs):

                prediction = self.predict(training_input.reshape(1, -1))

                error = training_output - prediction.flatten()

                self.weights += self.learning_rate * np.outer(training_input, error)
                self.bias += self.learning_rate * error

            print("We are at epoch: " + str(epoch))

        end_time = time.time()
        print('Training duration: %.2f seconds' % (end_time - start_time))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def one_hot_encode(labels, num_classes):
    num_examples = len(labels)
    one_hot_labels = np.zeros((num_examples, num_classes))
    for i in range(num_examples):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

def compute_accuracy(predictions, true_labels):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


if __name__ == "__main__":
    max_rss_megabytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    baseline = max_rss_megabytes / (1024.0 * 1024.0)
    print(f"Baseline is: {baseline:.2f} MB")



    num_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    max_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_megabytes = max_rss_bytes / (1024.0 * 1024.0)

    X_train = X_train.reshape(-1, 28 * 28) / 255.0  # Flatten and normalize
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)

    input_size = 784
    output_size = num_classes
    perceptron = Perceptron(input_size, output_size, softmax)

    perceptron.train(X_train, y_train_onehot)
    print(f"Memory used after training: {max_rss_megabytes - baseline:.2f} MB")

    predictions = perceptron.predict(X_test)
    accuracy = compute_accuracy(predictions, y_test_onehot)
    print("Accuracy on the testing set:", accuracy)