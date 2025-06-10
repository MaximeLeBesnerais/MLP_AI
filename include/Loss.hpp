#ifndef LOSS_HPP
#define LOSS_HPP

#include "Matrix.hpp"

class Loss
{
public:
    virtual ~Loss() = default;

    // Calculates the average loss for a batch
    virtual double calculate(const Matrix &y_pred, const Matrix &y_true) = 0;

    // Calculates the gradient of the loss with respect to the predictions
    virtual Matrix backward(const Matrix &y_pred, const Matrix &y_true) = 0;
};

#endif // LOSS_HPP
