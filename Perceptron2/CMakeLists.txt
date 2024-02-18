cmake_minimum_required(VERSION 3.27)
project(Perceptron)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_COMPILER g++)



# -ffast-math -spicy math, may not be good for this purpose
# -funroll-loops - this might result in a ridiculous executable size
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_executable(Perceptron main.cpp
        Perceptron.cpp
        Perceptron.h
        ParseMNIST.h
        Util.h
        Util.cpp
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Perceptron PUBLIC OpenMP::OpenMP_CXX)
endif()

