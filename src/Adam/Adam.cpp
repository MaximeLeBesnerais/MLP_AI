#include "Adam.hpp"
#include <cmath>

Adam::Adam(std::vector<DenseLayer> &layers, double learning_rate,
           double beta1, double beta2, double epsilon)
    : Optimizer(layers, learning_rate),
      m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_t(0)
{

    for (auto &layer : m_layers)
    {
        m_m_weights.emplace_back(layer.getWeights().getRows(), layer.getWeights().getCols());
        m_v_weights.emplace_back(layer.getWeights().getRows(), layer.getWeights().getCols());
        m_m_biases.emplace_back(layer.getBiases().getRows(), layer.getBiases().getCols());
        m_v_biases.emplace_back(layer.getBiases().getRows(), layer.getBiases().getCols());
    }
}

void Adam::step()
{
    m_t++;

    for (size_t i = 0; i < m_layers.size(); ++i)
    {
        // --- Update Weights ---
        Matrix &weights = m_layers[i].getWeights();
        const Matrix &d_weights = m_layers[i].getWeightsGradient();

        // m = beta1 * m + (1 - beta1) * g
        m_m_weights[i] = m_m_weights[i] * m_beta1 + d_weights * (1.0 - m_beta1);
        // v = beta2 * v + (1 - beta2) * g^2
        Matrix d_weights_sq = d_weights;
        d_weights_sq.element_multiply(d_weights);
        m_v_weights[i] = m_v_weights[i] * m_beta2 + d_weights_sq * (1.0 - m_beta2);

        // Bias correction
        Matrix m_hat = m_m_weights[i] * (1.0 / (1.0 - std::pow(m_beta1, m_t)));
        Matrix v_hat = m_v_weights[i] * (1.0 / (1.0 - std::pow(m_beta2, m_t)));

        // v_hat_sqrt = sqrt(v_hat) + epsilon
        v_hat.element_sqrt();
        v_hat.map([this](double val)
                  { return val + m_epsilon; });

        // Update = m_hat / v_hat_sqrt
        m_hat.element_divide(v_hat);
        weights.update(m_hat, m_learning_rate);

        // --- Update Biases (same logic) ---
        Matrix &biases = m_layers[i].getBiases();
        const Matrix &d_biases = m_layers[i].getBiasesGradient();

        m_m_biases[i] = m_m_biases[i] * m_beta1 + d_biases * (1.0 - m_beta1);
        Matrix d_biases_sq = d_biases;
        d_biases_sq.element_multiply(d_biases);
        m_v_biases[i] = m_v_biases[i] * m_beta2 + d_biases_sq * (1.0 - m_beta2);

        Matrix m_hat_b = m_m_biases[i] * (1.0 / (1.0 - std::pow(m_beta1, m_t)));
        Matrix v_hat_b = m_v_biases[i] * (1.0 / (1.0 - std::pow(m_beta2, m_t)));

        v_hat_b.element_sqrt();
        v_hat_b.map([this](double val)
                    { return val + m_epsilon; });

        m_hat_b.element_divide(v_hat_b);
        biases.update(m_hat_b, m_learning_rate);
    }
}