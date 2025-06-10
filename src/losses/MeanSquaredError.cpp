#include "losses/MeanSquaredError.hpp"
#include <cmath>
#include <stdexcept>

double MeanSquaredError::calculate(const Matrix &y_pred, const Matrix &y_true)
{
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols())
    {
        throw std::invalid_argument("Prediction and true value matrices must have the same dimensions.");
    }

    Matrix diff = y_pred - y_true;
    double sum_sq_err = 0.0;
    for (int i = 0; i < diff.getRows(); ++i)
    {
        for (int j = 0; j < diff.getCols(); ++j)
        {
            sum_sq_err += std::pow(diff(i, j), 2);
        }
    }

    return sum_sq_err / y_pred.getRows();
}

Matrix MeanSquaredError::backward(const Matrix &y_pred, const Matrix &y_true)
{
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols())
    {
        throw std::invalid_argument("Prediction and true value matrices must have the same dimensions.");
    }

    Matrix grad = y_pred - y_true;
    double normalizer = 2.0 / y_pred.getRows();
    return grad * normalizer;
}