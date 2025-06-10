#include "regularizers/L2Regularizer.hpp"
#include <cmath>

L2Regularizer::L2Regularizer(double lambda) : m_lambda(lambda) {}

double L2Regularizer::loss(const Matrix &weights)
{
    double sum_sq = 0.0;
    for (int i = 0; i < weights.getRows(); ++i)
    {
        for (int j = 0; j < weights.getCols(); ++j)
        {
            sum_sq += std::pow(weights(i, j), 2);
        }
    }
    return 0.5 * m_lambda * sum_sq;
}

Matrix L2Regularizer::gradient(const Matrix &weights)
{
    // Gradient is lambda * weights
    return weights * m_lambda;
}