#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizers/Optimizer.hpp"

class Adam : public Optimizer
{
public:
    Adam(std::vector<DenseLayer> &layers, double learning_rate = 0.001,
         double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void step() override;

private:
    double m_beta1;
    double m_beta2;
    double m_epsilon;
    int m_t; // Timestep

    // State for moving averages
    std::vector<Matrix> m_m_weights, m_v_weights;
    std::vector<Matrix> m_m_biases, m_v_biases;
};

#endif // ADAM_HPP