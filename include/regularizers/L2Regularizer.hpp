#ifndef L2_REGULARIZER_HPP
#define L2_REGULARIZER_HPP

#include "regularizers/Regularizer.hpp"

class L2Regularizer : public Regularizer
{
public:
    L2Regularizer(double lambda);
    double loss(const Matrix &weights) override;
    Matrix gradient(const Matrix &weights) override;

private:
    double m_lambda;
};

#endif // L2_REGULARIZER_HPP