#ifndef MODEL_HPP
#define MODEL_HPP

#include "layers/DenseLayer.hpp"
#include <vector>

class Model
{
public:
    Model();

    void add(DenseLayer layer);
    void backward(const Matrix &d_output);
    Matrix predict(Matrix input);

    std::vector<DenseLayer> &getLayers();

    void save(const std::string &filename) const;
    void load(const std::string &filename);

private:
    std::vector<DenseLayer> m_layers;
};

#endif // MODEL_HPP