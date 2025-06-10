#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>

class Matrix
{
public:
    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>> &data);

    int getRows() const;
    int getCols() const;

    double &operator()(int r, int c);
    const double &operator()(int r, int c) const;

    static Matrix random(int rows, int cols);
    static Matrix multiply(const Matrix &a, const Matrix &b);
    static Matrix he(int rows, int cols);

    Matrix operator-(const Matrix &other) const;
    Matrix operator*(double scalar) const;
    Matrix operator+(const Matrix &other) const;

    void element_multiply(const Matrix &other);
    void element_divide(const Matrix &other);
    void element_sqrt();

    Matrix transpose() const;
    void map(const std::function<double(double)> &func);
    Matrix slice(int start_row, int end_row) const;

    void update(const Matrix &gradient, double learning_rate);

    void print() const;

private:
    int m_rows;
    int m_cols;
    std::vector<double> m_data;
};

#endif // MATRIX_H
