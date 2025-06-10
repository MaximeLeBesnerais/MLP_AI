#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "Matrix.hpp"
#include "Activation.hpp"
#include "Regularizer.hpp"
#include <memory>

enum class WeightInitType
{
    RANDOM, // Our original method
    HE      // The new, better method for ReLU
};

class DenseLayer
{
public:
    DenseLayer(int inputSize, int outputSize, std::shared_ptr<Activation> activation,
               std::shared_ptr<Regularizer> regularizer = nullptr,
               WeightInitType init_type = WeightInitType::HE);

    Matrix forward(const Matrix &inputData);
    Matrix backward(const Matrix &d_output);

    // Getters
    Matrix &getWeights();
    Matrix &getBiases();
    // Const versions for read-only access
    const Matrix &getWeights() const;
    const Matrix &getBiases() const;
    
    const Matrix &getInput() const;
    std::shared_ptr<Activation> getActivation() const;
    const Matrix &getWeightsGradient() const;
    const Matrix &getBiasesGradient() const;
    std::shared_ptr<Regularizer> getRegularizer() const;

    // Setters
    void setWeights(const Matrix &weights);
    void setBiases(const Matrix &biases);

private:
    Matrix m_weights;
    Matrix m_biases;
    std::shared_ptr<Activation> m_activation;
    Matrix m_input;
    std::shared_ptr<Regularizer> m_regularizer;

    Matrix m_d_weights;
    Matrix m_d_biases;
};

#endif // DENSE_LAYER_HPP
