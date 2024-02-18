
#include <iostream>
#include <vector>
#include <chrono>

#include "ParseMNIST.h"
#include "Perceptron.h"
#include "Util.h"

using namespace std::chrono;

double test(const std::vector<std::vector<double>>& test_inputs, const std::vector<size_t>& test_labels, const Perceptron& perceptron) {
    int correct_predictions = 0;

    for (int example_index = 0; example_index < test_inputs.size(); ++example_index) {
        auto predictions = perceptron.predict(test_inputs[example_index]);

        int max_index = 0;
        for (int i = 0; i < 10; ++i) {
            if (predictions[i] > predictions[max_index]) {
                max_index = i;
            }
        }

        if (max_index == test_labels[example_index]) {
            correct_predictions++;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / test_inputs.size();
    return accuracy;
}




int main() {
    baselineMemory =  getMemoryUsage();
    std::cout << "Baseline memory: " << baselineMemory << '\n';

    std::vector<std::vector<double>> test_images;
    std::vector<size_t> test_labels;
    loadMNISTData("files/mnist_test.csv", test_images, test_labels, 10000);

    std::vector<std::vector<double>> train_images;
    std::vector<size_t> train_labels;
    loadMNISTData("files/mnist_train.csv", train_images, train_labels, 60000);
    std::cout << "Memory used after loading the dataset: " << getMemoryUsage() - baselineMemory << '\n';

    int input_size = 784;
    int output_size = 10;
    double learning_rate = 0.01;
    int epochs = 10;

    Perceptron perceptron(input_size, output_size, learning_rate);

    auto start = high_resolution_clock::now();
    perceptron.train(train_images, train_labels, epochs);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Training duration: " << duration.count()/1000.0 << '\n';
    std::cout << "Memory used at the end point: " << getMemoryUsage() - baselineMemory << '\n';

    std::cout << "Perceptron accuracy: " << test(test_images, test_labels, perceptron );


    return 0;
}
