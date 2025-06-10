#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include "Matrix.hpp"
#include <vector>

// Helper to convert probability matrix to a matrix of predicted class indices
Matrix get_predictions(const Matrix &y_pred);

// Calculates classification accuracy
double calculate_accuracy(const Matrix &y_pred, const Matrix &y_true_raw);

// Class to compute and display a confusion matrix
class ConfusionMatrix
{
public:
    ConfusionMatrix(int num_classes);
    void update(const Matrix &y_pred, const Matrix &y_true_raw);
    void print() const;

private:
    int m_num_classes;
    Matrix m_matrix;
};

#endif // EVALUATION_HPP