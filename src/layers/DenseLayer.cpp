#include "DenseLayer.hpp"
#include <stdexcept>

DenseLayer::DenseLayer(int inputSize, int outputSize, std::shared_ptr<Activation> activation,
                       std::shared_ptr<Regularizer> regularizer,
                       WeightInitType init_type)
    : m_weights(Matrix::random(inputSize, outputSize)),
      m_biases(Matrix::random(1, outputSize)),
      m_activation(activation),
      m_input(0, 0), // Initialize m_input before m_regularizer
      m_regularizer(regularizer),
      m_d_weights(inputSize, outputSize),
      m_d_biases(1, outputSize)
{
    switch (init_type)
    {
    case WeightInitType::HE:
        m_weights = Matrix::he(inputSize, outputSize);
        break;
    case WeightInitType::RANDOM:
    default:
        m_weights = Matrix::random(inputSize, outputSize);
        break;
    }
}

Matrix DenseLayer::backward(const Matrix &d_output)
{
    Matrix d_linear = m_activation->backward(d_output);
    m_d_weights = Matrix::multiply(m_input.transpose(), d_linear);

    if (m_regularizer)
    {
        m_d_weights = m_d_weights + m_regularizer->gradient(m_weights);
    }
    for (int j = 0; j < m_d_biases.getCols(); ++j)
    {
        double sum = 0.0;
        for (int i = 0; i < d_linear.getRows(); ++i)
        {
            sum += d_linear(i, j);
        }
        m_d_biases(0, j) = sum;
    }

    Matrix d_input = Matrix::multiply(d_linear, m_weights.transpose());
    return d_input;
}

const Matrix &DenseLayer::getWeightsGradient() const { return m_d_weights; }
const Matrix &DenseLayer::getBiasesGradient() const { return m_d_biases; }

Matrix &DenseLayer::getWeights() { return m_weights; }
Matrix &DenseLayer::getBiases() { return m_biases; }
const Matrix &DenseLayer::getWeights() const { return m_weights; }
const Matrix &DenseLayer::getBiases() const { return m_biases; }

const Matrix &DenseLayer::getInput() const { return m_input; }
std::shared_ptr<Activation> DenseLayer::getActivation() const { return m_activation; }
std::shared_ptr<Regularizer> DenseLayer::getRegularizer() const { return m_regularizer; }

void DenseLayer::setWeights(const Matrix& weights) {
    if (m_weights.getRows() != weights.getRows() || m_weights.getCols() != weights.getCols()) {
        throw std::invalid_argument("New weights matrix has incorrect dimensions.");
    }
    m_weights = weights;
}

void DenseLayer::setBiases(const Matrix& biases) {
    if (m_biases.getRows() != biases.getRows() || m_biases.getCols() != biases.getCols()) {
        throw std::invalid_argument("New biases matrix has incorrect dimensions.");
    }
    m_biases = biases;
}

Matrix DenseLayer::forward(const Matrix &inputData)
{
    m_input = inputData; // Store a copy of the input

    Matrix z = Matrix::multiply(inputData, m_weights);
    for (int i = 0; i < z.getRows(); ++i)
    {
        for (int j = 0; j < z.getCols(); ++j)
        {
            z(i, j) += m_biases(0, j);
        }
    }

    return m_activation->forward(z);
}