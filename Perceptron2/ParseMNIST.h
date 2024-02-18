#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


// Function to load MNIST data from CSV files
void loadMNISTData(const std::string& filename, std::vector<std::vector<double>>& imagesOut, std::vector<size_t>& labelsOut, int num_samples) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << '\n';
        return;
    }

    imagesOut.resize(num_samples, std::vector<double>(784));
    labelsOut.resize(num_samples);

    // Skip the first line
    std::string first_line;
    std::getline(file, first_line);

    std::string line;
    int count = 0;
    while (std::getline(file, line) && count < num_samples) {
        std::istringstream iss(line);
        std::string token;

        std::getline(iss, token, ',');

        //std::cout << "token: " << token << '\n';

        labelsOut[count] = std::stoi(token);

        int pixel_index = 0;
        while (std::getline(iss, token, ',')) {
            imagesOut[count][pixel_index] = std::stod(token) / 255.0;
            pixel_index++;
        }
        count++;
    }

    std::cout << "Final count: " << count << '\n';
    file.close();
}
