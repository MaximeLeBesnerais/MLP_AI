#ifndef MEAN_SQUARED_ERROR_HPP
#define MEAN_SQUARED_ERROR_HPP

#include "losses/Loss.hpp"

class MeanSquaredError : public Loss {
public:
    double calculate(const Matrix& y_pred, const Matrix& y_true) override;
    Matrix backward(const Matrix& y_pred, const Matrix& y_true) override;
};

#endif // MEAN_SQUARED_ERROR_HPP