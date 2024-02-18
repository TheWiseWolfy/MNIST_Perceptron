#include "Util.h"


double getMemoryUsage(){
    struct rusage memInfo;

    getrusage(RUSAGE_SELF, &memInfo); //This returns in kb unlike in python

    double memoryInMB = (double)memInfo.ru_maxrss / 1024.0;
    return memoryInMB;
}

std::vector<std::vector<bool>> one_hot_encode(const std::vector<size_t>& labels, size_t num_classes) {
    std::vector<std::vector<bool>> one_hot_encoded;

    for (size_t label : labels) {
        std::vector<bool> one_hot(num_classes, false);

        if (label < num_classes) {
            one_hot[label] = true;
        }

        one_hot_encoded.push_back(one_hot);
    }

    return one_hot_encoded;
}

