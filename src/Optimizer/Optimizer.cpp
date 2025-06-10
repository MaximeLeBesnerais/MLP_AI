#include "Optimizer.hpp"

Optimizer::Optimizer(std::vector<DenseLayer> &layers, double learning_rate)
    : m_layers(layers), m_learning_rate(learning_rate) {}