cmake_minimum_required(VERSION 3.27)
project(Perceptron)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_COMPILER g++)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")

add_executable(Perceptron main.cpp
        Perceptron.cpp
        Perceptron.h
        ParseMNIST.h
        Util.h
        Util.cpp
)
