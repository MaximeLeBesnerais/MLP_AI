#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "layers/DenseLayer.hpp"
#include <vector>

class Optimizer
{
public:
    Optimizer(std::vector<DenseLayer> &layers, double learning_rate);
    virtual ~Optimizer() = default;

    virtual void step() = 0;

protected:
    std::vector<DenseLayer> &m_layers;
    double m_learning_rate;
};

#endif // OPTIMIZER_HPP