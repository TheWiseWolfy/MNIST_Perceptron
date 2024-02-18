#pragma once

#include <vector>

#include "sys/resource.h"

static double baselineMemory;

double getMemoryUsage();
std::vector<std::vector<bool>> one_hot_encode(const std::vector<size_t>& labels, size_t num_classes);

