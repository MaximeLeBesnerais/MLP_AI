#ifndef ELASTIC_NET_REGULARIZER_HPP
#define ELASTIC_NET_REGULARIZER_HPP

#include "regularizers/Regularizer.hpp"

class ElasticNetRegularizer : public Regularizer
{
public:
    // lambda1 for L1, lambda2 for L2
    ElasticNetRegularizer(double lambda1, double lambda2);
    double loss(const Matrix &weights) override;
    Matrix gradient(const Matrix &weights) override;

private:
    double m_lambda1;
    double m_lambda2;
};

#endif // ELASTIC_NET_REGULARIZER_HPP