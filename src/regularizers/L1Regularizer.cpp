#include "regularizers/L1Regularizer.hpp"
#include <cmath>

L1Regularizer::L1Regularizer(double lambda) : m_lambda(lambda) {}

double L1Regularizer::loss(const Matrix &weights)
{
    double sum_abs = 0.0;
    for (int i = 0; i < weights.getRows(); ++i)
    {
        for (int j = 0; j < weights.getCols(); ++j)
        {
            sum_abs += std::abs(weights(i, j));
        }
    }
    return m_lambda * sum_abs;
}

Matrix L1Regularizer::gradient(const Matrix &weights)
{
    Matrix grad = weights;
    grad.map([this](double w)
             {
        if (w > 0) return m_lambda;
        if (w < 0) return -m_lambda;
        return 0.0; });
    return grad;
}