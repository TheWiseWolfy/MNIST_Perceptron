#pragma once

#include <vector>
#include <cmath>
#include <unordered_map>
#include <iostream>

#include "Util.h"

class Perceptron {
private:
    std::vector<std::vector<double>> weights; // Weights for each perceptron
    std::vector<double> biases; // Biases for each perceptron
    double learning_rate;

public:
    Perceptron(int input_size, int output_size, double learning_rate) : learning_rate(learning_rate) {
        weights.resize(output_size, std::vector<double>(input_size, 0.0));
        biases.resize(output_size, 0.0);

        std::cout << "Memory used in perceptron: " << getMemoryUsage() - baselineMemory  << '\n';
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }


     std::unordered_map<int, double> predict(const std::vector<double>& inputs) const{
        std::unordered_map<int, double> predictions;

        for (int perceptron_index = 0; perceptron_index < biases.size(); ++perceptron_index) {
            double weighted_sum = biases[perceptron_index];

            for (int input_index = 0; input_index < inputs.size(); ++input_index) {
                weighted_sum += weights[perceptron_index][input_index] * inputs[input_index];
            }

            predictions[perceptron_index] = sigmoid(weighted_sum);
        }

        return predictions;
    }

    void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<size_t>& labels, int epochs) {

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int example_index = 0; example_index < training_inputs.size(); ++example_index) {

                auto predictions = predict(training_inputs[example_index]);

                auto one_hot_label = [&labels, &example_index, this](int perceptron_index) {
                    std::vector<double> one_hot(biases.size(), 0.0);
                    one_hot[labels[example_index]] = 1.0;
                    return one_hot[perceptron_index];
                };

                for (int perceptron_index = 0; perceptron_index < weights.size(); ++perceptron_index) {
                    double error = one_hot_label(perceptron_index) - predictions[perceptron_index];

                    biases[perceptron_index] += learning_rate * error;

                    for (int weight_index = 0; weight_index < weights[perceptron_index].size(); ++weight_index) {
                        weights[perceptron_index][weight_index] += learning_rate * error * training_inputs[example_index][weight_index];
                    }
                }
            }

            std::cout << "We finished epoch: " << epoch << '\n';
        }
    }


};

