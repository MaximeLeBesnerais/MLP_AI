#ifndef REGULARIZER_HPP
#define REGULARIZER_HPP

#include "Matrix.hpp"

class Regularizer
{
public:
    virtual ~Regularizer() = default;
    virtual double loss(const Matrix &weights) = 0;
    virtual Matrix gradient(const Matrix &weights) = 0;
};

#endif // REGULARIZER_HPP