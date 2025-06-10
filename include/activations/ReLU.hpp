#ifndef RELU_HPP
#define RELU_HPP

#include "Activation.hpp"

class ReLU : public Activation
{
public:
    ReLU();
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &d_output) override;

private:
    Matrix m_input;
};

#endif // RELU_HPP
