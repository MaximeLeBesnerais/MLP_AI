#include "DataHandler.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath> // For std::sqrt and std::pow

// Add this class implementation to DataHandler.cpp
#include <stdexcept> // For StandardScaler

StandardScaler::StandardScaler() : m_mean(0, 0), m_std(0, 0) {}

void StandardScaler::fit(const Matrix &data)
{
    int rows = data.getRows();
    int cols = data.getCols();
    if (rows == 0)
        return;

    m_mean = Matrix(1, cols);
    m_std = Matrix(1, cols);

    for (int j = 0; j < cols; ++j)
    {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i)
        {
            sum += data(i, j);
        }
        m_mean(0, j) = sum / rows;

        double sum_sq_diff = 0.0;
        for (int i = 0; i < rows; ++i)
        {
            sum_sq_diff += std::pow(data(i, j) - m_mean(0, j), 2);
        }
        m_std(0, j) = std::sqrt(sum_sq_diff / rows);
        if (m_std(0, j) == 0)
            m_std(0, j) = 1.0; // Avoid division by zero
    }
}

Matrix StandardScaler::transform(const Matrix &data) const
{
    if (data.getCols() != m_mean.getCols())
    {
        throw std::runtime_error("Data has incorrect number of features for transform.");
    }
    Matrix scaled_data = data;
    for (int j = 0; j < data.getCols(); ++j)
    {
        for (int i = 0; i < data.getRows(); ++i)
        {
            scaled_data(i, j) = (scaled_data(i, j) - m_mean(0, j)) / m_std(0, j);
        }
    }
    return scaled_data;
}

Matrix StandardScaler::fit_transform(const Matrix &data)
{
    fit(data);
    return transform(data);
}

std::pair<Matrix, Matrix> read_csv_mnist(const std::string &filepath, int num_rows)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<std::vector<double>> features_vec;
    std::vector<std::vector<double>> labels_vec;
    std::string line;

    // Skip header
    std::getline(file, line);

    int rows_read = 0;
    while (std::getline(file, line) && (num_rows == -1 || rows_read < num_rows))
    {
        std::stringstream ss(line);
        std::string cell;

        // First column is the label
        std::getline(ss, cell, ',');
        labels_vec.push_back({std::stod(cell)});

        // The rest are features
        std::vector<double> feature_row;
        while (std::getline(ss, cell, ','))
        {
            feature_row.push_back(std::stod(cell));
        }
        features_vec.push_back(feature_row);
        rows_read++;
    }

    return {Matrix(features_vec), Matrix(labels_vec)};
}

Matrix read_csv_boston(const std::string &filepath, int num_rows)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<std::vector<double>> data_vec;
    std::string line;

    std::getline(file, line); // Skip header

    int rows_read = 0;
    while (std::getline(file, line) && (num_rows == -1 || rows_read < num_rows))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row_vec;

        while (std::getline(ss, cell, ','))
        {
            try
            {
                row_vec.push_back(std::stod(cell));
            }
            catch (const std::invalid_argument &)
            {
                row_vec.push_back(0.0); // Handle "NA"
            }
        }
        data_vec.push_back(row_vec);
        rows_read++;
    }

    return Matrix(data_vec);
}

void normalize_features(Matrix &features)
{
    features.map([](double val)
                 { return val / 255.0; });
}

Matrix one_hot_encode(const Matrix &labels, int num_classes)
{
    Matrix one_hot(labels.getRows(), num_classes);
    for (int i = 0; i < labels.getRows(); ++i)
    {
        int label_val = static_cast<int>(labels(i, 0));
        if (label_val >= 0 && label_val < num_classes)
        {
            one_hot(i, label_val) = 1.0;
        }
    }
    return one_hot;
}
