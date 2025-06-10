#ifndef LINEAR_ACTIVATION_HPP
#define LINEAR_ACTIVATION_HPP

#include "Activation.hpp"

class LinearActivation : public Activation
{
public:
    LinearActivation();
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &d_output) override;
};

#endif // LINEAR_ACTIVATION_HPP