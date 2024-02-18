#pragma once

#include <vector>
#include <cmath>
#include <unordered_map>
#include <iostream>

#include <omp.h>

#include "Util.h"

class Perceptron {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
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

    std::vector<double> predict(const std::vector<double>& inputs) const{
        std::vector<double> local_predictions(biases.size());

        //10 iterations loop,
        #pragma omp parallel for schedule(static)
        for (int perceptron_index = 0; perceptron_index < biases.size(); ++perceptron_index) {
            double weighted_sum = biases[perceptron_index];

            //784 iterations loop
            for (int input_index = 0; input_index < inputs.size(); ++input_index) {
                weighted_sum += weights[perceptron_index][input_index] * inputs[input_index];
            }

            local_predictions[perceptron_index] = sigmoid(weighted_sum);
        }

        return local_predictions;
    }

    void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<bool>>& one_hot_labels, int epochs) {

        // Iterative process
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Iterative process once again
            for (int example_index = 0; example_index < training_inputs.size(); ++example_index) {

                auto predictions = predict(training_inputs[example_index]);

                #pragma omp parallel for schedule(static)
                for (int perceptron_index = 0; perceptron_index < weights.size(); ++perceptron_index) {
                    double error = one_hot_labels[example_index][perceptron_index] - predictions[perceptron_index];

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

