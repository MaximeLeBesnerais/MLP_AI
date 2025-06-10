#ifndef L1_REGULARIZER_HPP
#define L1_REGULARIZER_HPP

#include "Regularizer.hpp"

class L1Regularizer : public Regularizer
{
public:
    L1Regularizer(double lambda);
    double loss(const Matrix &weights) override;
    Matrix gradient(const Matrix &weights) override;

private:
    double m_lambda;
};

#endif // L1_REGULARIZER_HPP