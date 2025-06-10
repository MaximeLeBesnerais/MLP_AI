#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Matrix.hpp"

class Activation
{
public:
    virtual ~Activation() = default;
    virtual Matrix forward(const Matrix &input) = 0;
    virtual Matrix backward(const Matrix &d_output) = 0;
};

#endif // ACTIVATION_HPP