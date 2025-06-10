#ifndef CATEGORICAL_CROSS_ENTROPY_HPP
#define CATEGORICAL_CROSS_ENTROPY_HPP

#include "Loss.hpp"

class CategoricalCrossEntropy : public Loss {
public:
    double calculate(const Matrix& y_pred, const Matrix& y_true) override;
    Matrix backward(const Matrix& y_pred, const Matrix& y_true) override;
};

#endif // CATEGORICAL_CROSS_ENTROPY_HPP