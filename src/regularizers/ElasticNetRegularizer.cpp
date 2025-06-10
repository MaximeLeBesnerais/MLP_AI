#include "ElasticNetRegularizer.hpp"
#include <cmath>

ElasticNetRegularizer::ElasticNetRegularizer(double lambda1, double lambda2)
    : m_lambda1(lambda1), m_lambda2(lambda2) {}

double ElasticNetRegularizer::loss(const Matrix &weights)
{
    // L1 loss part
    double l1_loss = 0.0;
    for (int i = 0; i < weights.getRows(); ++i)
    {
        for (int j = 0; j < weights.getCols(); ++j)
        {
            l1_loss += std::abs(weights(i, j));
        }
    }
    l1_loss *= m_lambda1;

    // L2 loss part
    double l2_loss = 0.0;
    for (int i = 0; i < weights.getRows(); ++i)
    {
        for (int j = 0; j < weights.getCols(); ++j)
        {
            l2_loss += std::pow(weights(i, j), 2);
        }
    }
    l2_loss *= 0.5 * m_lambda2;

    return l1_loss + l2_loss;
}

Matrix ElasticNetRegularizer::gradient(const Matrix &weights)
{
    // L1 gradient part
    Matrix l1_grad = weights;
    l1_grad.map([this](double w)
                {
        if (w > 0) return m_lambda1;
        if (w < 0) return -m_lambda1;
        return 0.0; });

    // L2 gradient part
    Matrix l2_grad = weights * m_lambda2;

    return l1_grad + l2_grad;
}