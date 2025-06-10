#include "Matrix.hpp"
#include <stdexcept>
#include <random>
#include <cmath>

Matrix::Matrix(int rows, int cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0) {}

Matrix::Matrix(const std::vector<std::vector<double>> &data)
{
    if (data.empty() || data[0].empty())
    {
        m_rows = 0;
        m_cols = 0;
        return;
    }
    m_rows = data.size();
    m_cols = data[0].size();
    m_data.resize(m_rows * m_cols);
    for (int i = 0; i < m_rows; ++i)
    {
        for (int j = 0; j < m_cols; ++j)
        {
            m_data[i * m_cols + j] = data[i][j];
        }
    }
}

int Matrix::getRows() const
{
    return m_rows;
}

int Matrix::getCols() const
{
    return m_cols;
}

double &Matrix::operator()(int r, int c)
{
    if (r >= m_rows || c >= m_cols || r < 0 || c < 0)
    {
        throw std::out_of_range("Matrix index out of range");
    }
    return m_data[r * m_cols + c];
}

const double &Matrix::operator()(int r, int c) const
{
    if (r >= m_rows || c >= m_cols || r < 0 || c < 0)
    {
        throw std::out_of_range("Matrix index out of range");
    }
    return m_data[r * m_cols + c];
}

Matrix Matrix::random(int rows, int cols)
{
    Matrix m(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            m(i, j) = dis(gen);
        }
    }
    return m;
}

Matrix Matrix::multiply(const Matrix &a, const Matrix &b)
{
    if (a.getCols() != b.getRows())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    Matrix result(a.getRows(), b.getCols());
    for (int i = 0; i < a.getRows(); ++i)
    {
        for (int j = 0; j < b.getCols(); ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < a.getCols(); ++k)
            {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::he(int rows, int cols)
{
    Matrix m(rows, cols);
    double stddev = std::sqrt(2.0 / rows);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, stddev);

    for (int i = 0; i < rows * cols; ++i)
    {
        m.m_data[i] = dis(gen);
    }
    return m;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (m_rows != other.m_rows || m_cols != other.m_cols)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
    }
    Matrix result(m_rows, m_cols);
    for (size_t i = 0; i < m_data.size(); ++i)
    {
        result.m_data[i] = m_data[i] - other.m_data[i];
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(m_rows, m_cols);
    for (size_t i = 0; i < m_data.size(); ++i)
    {
        result.m_data[i] = m_data[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (m_rows != other.m_rows || m_cols != other.m_cols)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    Matrix result(m_rows, m_cols);
    for (size_t i = 0; i < m_data.size(); ++i)
    {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    return result;
}

void Matrix::element_multiply(const Matrix &other)
{
    if (m_rows != other.m_rows || m_cols != other.m_cols)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for element-wise multiplication.");
    }
    for (size_t i = 0; i < m_data.size(); ++i)
    {
        m_data[i] *= other.m_data[i];
    }
}

void Matrix::element_divide(const Matrix &other)
{
    if (m_rows != other.m_rows || m_cols != other.m_cols)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for element-wise division.");
    }
    for (size_t i = 0; i < m_data.size(); ++i)
    {
        if (other.m_data[i] == 0)
        {
            // Avoid division by zero, though Adam's epsilon helps
            m_data[i] = 0;
        }
        else
        {
            m_data[i] /= other.m_data[i];
        }
    }
}

void Matrix::element_sqrt()
{
    for (size_t i = 0; i < m_data.size(); ++i)
    {
        m_data[i] = std::sqrt(m_data[i]);
    }
}

Matrix Matrix::transpose() const
{
    Matrix result(m_cols, m_rows);
    for (int i = 0; i < m_rows; ++i)
    {
        for (int j = 0; j < m_cols; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Matrix::map(const std::function<double(double)> &func)
{
    for (int i = 0; i < m_rows * m_cols; ++i)
    {
        m_data[i] = func(m_data[i]);
    }
}

Matrix Matrix::slice(int start_row, int end_row) const
{
    if (start_row < 0 || end_row > m_rows || start_row >= end_row)
    {
        throw std::out_of_range("Invalid row range for slice.");
    }
    int num_sliced_rows = end_row - start_row;
    Matrix sliced(num_sliced_rows, m_cols);
    for (int i = 0; i < num_sliced_rows; ++i)
    {
        for (int j = 0; j < m_cols; ++j)
        {
            sliced(i, j) = (*this)(start_row + i, j);
        }
    }
    return sliced;
}

void Matrix::update(const Matrix &gradient, double learning_rate)
{
    if (m_rows != gradient.m_rows || m_cols != gradient.m_cols)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for update.");
    }

    for (size_t i = 0; i < m_data.size(); ++i)
    {
        m_data[i] -= gradient.m_data[i] * learning_rate;
    }
}

void Matrix::print() const
{
    for (int i = 0; i < m_rows; ++i)
    {
        for (int j = 0; j < m_cols; ++j)
        {
            std::cout << (*this)(i, j) << "\t";
        }
        std::cout << std::endl;
    }
}