#ifndef SGD_HPP
#define SGD_HPP

#include "optimizers/Optimizer.hpp"

class SGD : public Optimizer
{
public:
    SGD(std::vector<DenseLayer> &layers, double learning_rate = 0.01);
    void step() override;
};

#endif // SGD_HPP