#include "Model.hpp"
#include <fstream>
#include <sstream>
#include <string>

Model::Model() {}

void Model::add(DenseLayer layer)
{
    m_layers.push_back(layer);
}

void Model::backward(const Matrix &d_output)
{
    Matrix current_grad = d_output;
    for (int i = m_layers.size() - 1; i >= 0; --i)
    {
        current_grad = m_layers[i].backward(current_grad);
    }
}

std::vector<DenseLayer> &Model::getLayers()
{
    return m_layers;
}

Matrix Model::predict(Matrix input)
{
    Matrix current_output = input;
    for (auto &layer : m_layers)
    {
        current_output = layer.forward(current_output);
    }
    return current_output;
}

void Model::save(const std::string &filepath) const
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for saving: " + filepath);
    }

    for (const auto &layer : m_layers)
    {
        const Matrix &weights = layer.getWeights();
        const Matrix &biases = layer.getBiases();

        // Save weights
        file << "WEIGHTS\n";
        file << weights.getRows() << "," << weights.getCols() << "\n";
        for (int i = 0; i < weights.getRows(); ++i)
        {
            for (int j = 0; j < weights.getCols(); ++j)
            {
                file << weights(i, j) << (j == weights.getCols() - 1 ? "" : ",");
            }
            file << "\n";
        }

        // Save biases
        file << "BIASES\n";
        file << biases.getRows() << "," << biases.getCols() << "\n";
        for (int i = 0; i < biases.getRows(); ++i)
        {
            for (int j = 0; j < biases.getCols(); ++j)
            {
                file << biases(i, j) << (j == biases.getCols() - 1 ? "" : ",");
            }
            file << "\n";
        }
    }
    file.close();
}

void Model::load(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for loading: " + filepath);
    }

    std::string line;
    size_t layer_idx = 0;
    while (layer_idx < m_layers.size() && std::getline(file, line))
    {
        if (line != "WEIGHTS")
            continue;

        // Load Weights
        std::getline(file, line); // Get dimensions
        std::stringstream ss_dims(line);
        std::string rows_str, cols_str;
        std::getline(ss_dims, rows_str, ',');
        std::getline(ss_dims, cols_str, ',');
        int rows = std::stoi(rows_str);
        int cols = std::stoi(cols_str);

        Matrix weights(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            std::getline(file, line);
            std::stringstream ss_row(line);
            std::string val_str;
            for (int j = 0; j < cols; ++j)
            {
                std::getline(ss_row, val_str, ',');
                weights(i, j) = std::stod(val_str);
            }
        }
        m_layers[layer_idx].setWeights(weights);

        // Load Biases
        std::getline(file, line); // Should be "BIASES"
        if (line != "BIASES")
            break; // Or throw error

        std::getline(file, line); // Get dimensions
        std::stringstream ss_bdims(line);
        std::getline(ss_bdims, rows_str, ',');
        std::getline(ss_bdims, cols_str, ',');
        rows = std::stoi(rows_str);
        cols = std::stoi(cols_str);

        Matrix biases(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            std::getline(file, line);
            std::stringstream ss_brow(line);
            std::string val_str;
            for (int j = 0; j < cols; ++j)
            {
                std::getline(ss_brow, val_str, ',');
                biases(i, j) = std::stod(val_str);
            }
        }
        m_layers[layer_idx].setBiases(biases);

        layer_idx++;
    }
    file.close();
}
