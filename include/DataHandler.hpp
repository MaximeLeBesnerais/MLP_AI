#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include "Matrix.hpp"
#include <string>
#include <vector>

// Add this class declaration to DataHandler.hpp
class StandardScaler
{
public:
    StandardScaler();
    void fit(const Matrix &data);
    Matrix transform(const Matrix &data) const;
    Matrix fit_transform(const Matrix &data);

private:
    Matrix m_mean;
    Matrix m_std;
};

// Reads a CSV file, assuming the first column is the label.
std::pair<Matrix, Matrix> read_csv_mnist(const std::string &filepath, int num_rows = -1);

// Reads a CSV file for the Boston Housing dataset, Reads all columns into one matrix. Handles NA values.
Matrix read_csv_boston(const std::string &filepath, int num_rows = -1);

// Normalizes feature values from [0, 255] to [0, 1].
void normalize_features(Matrix &features);

// Converts a column vector of labels to a one-hot encoded matrix.
Matrix one_hot_encode(const Matrix &labels, int num_classes);

#endif // DATA_HANDLER_HPP