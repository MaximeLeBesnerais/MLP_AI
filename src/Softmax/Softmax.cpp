#include "Softmax.hpp"
#include <cmath>
#include <numeric>

Softmax::Softmax() {}

Matrix Softmax::forward(const Matrix &input)
{
    Matrix output(input.getRows(), input.getCols());
    for (int i = 0; i < input.getRows(); ++i)
    {
        // Find max value in the row for stability
        double max_val = input(i, 0);
        for (int j = 1; j < input.getCols(); ++j)
        {
            if (input(i, j) > max_val)
            {
                max_val = input(i, j);
            }
        }

        // Exponentiate and sum
        double sum = 0.0;
        for (int j = 0; j < input.getCols(); ++j)
        {
            output(i, j) = std::exp(input(i, j) - max_val);
            sum += output(i, j);
        }

        // Normalize to get probabilities
        for (int j = 0; j < input.getCols(); ++j)
        {
            output(i, j) /= sum;
        }
    }
    return output;
}

Matrix Softmax::backward(const Matrix &d_output)
{
    // As explained, the true Jacobian-vector product is complex.
    // We will use a simplified gradient calculation in the loss function itself.
    // This function will thus not be used in our final training loop for classification.
    // For now, we just pass the gradient through.
    return d_output;
}