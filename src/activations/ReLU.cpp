#include "activations/ReLU.hpp"

ReLU::ReLU() : m_input(0, 0) {}

Matrix ReLU::forward(const Matrix &input)
{
    m_input = input;
    Matrix output = input;
    output.map([](double val)
               { return val > 0 ? val : 0; });
    return output;
}

Matrix ReLU::backward(const Matrix &d_output)
{
    Matrix d_input = m_input;
    d_input.map([](double val)
                { return val > 0 ? 1.0 : 0.0; });

    // Element-wise multiplication
    for (int i = 0; i < d_input.getRows(); ++i)
    {
        for (int j = 0; j < d_input.getCols(); ++j)
        {
            d_input(i, j) *= d_output(i, j);
        }
    }
    return d_input;
}
