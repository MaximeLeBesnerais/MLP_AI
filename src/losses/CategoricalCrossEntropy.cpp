#include "losses/CategoricalCrossEntropy.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

double CategoricalCrossEntropy::calculate(const Matrix &y_pred, const Matrix &y_true)
{
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols())
    {
        throw std::invalid_argument("Prediction and true value matrices must have the same dimensions.");
    }

    int samples = y_pred.getRows();
    int classes = y_pred.getCols();
    double total_loss = 0.0;
    double epsilon = std::numeric_limits<double>::epsilon();

    for (int i = 0; i < samples; ++i)
    {
        for (int j = 0; j < classes; ++j)
        {
            // Clip predictions to avoid log(0)
            double pred_clipped = std::max(epsilon, y_pred(i, j));
            pred_clipped = std::min(1.0 - epsilon, pred_clipped);

            total_loss += y_true(i, j) * std::log(pred_clipped);
        }
    }

    return -total_loss / samples;
}

Matrix CategoricalCrossEntropy::backward(const Matrix &y_pred, const Matrix &y_true)
{
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols())
    {
        throw std::invalid_argument("Prediction and true value matrices must have the same dimensions.");
    }
    // This is the combined, simplified gradient of CCE+Softmax
    return y_pred - y_true;
}