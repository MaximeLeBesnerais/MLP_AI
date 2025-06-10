#include "optimizers/SGD.hpp"

SGD::SGD(std::vector<DenseLayer> &layers, double learning_rate)
    : Optimizer(layers, learning_rate) {}

void SGD::step()
{
    for (auto &layer : m_layers)
    {
        Matrix &weights = layer.getWeights();
        Matrix &biases = layer.getBiases();

        const Matrix &d_weights = layer.getWeightsGradient();
        const Matrix &d_biases = layer.getBiasesGradient();

        weights.update(d_weights, m_learning_rate);
        biases.update(d_biases, m_learning_rate);
    }
}