#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "Activation.hpp"

class Softmax : public Activation
{
public:
    Softmax();
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &d_output) override;
};

#endif // SOFTMAX_HPP