#include "LinearActivation.hpp"

LinearActivation::LinearActivation() {}

// Forward pass just returns the input
Matrix LinearActivation::forward(const Matrix& input) {
    return input;
}

// Backward pass just returns the gradient (derivative is 1)
Matrix LinearActivation::backward(const Matrix& d_output) {
    return d_output;
}