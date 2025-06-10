#include "Evaluation.hpp"
#include <iostream>
#include <iomanip>

Matrix get_predictions(const Matrix &y_pred)
{
    Matrix predictions(y_pred.getRows(), 1);
    for (int i = 0; i < y_pred.getRows(); ++i)
    {
        double max_val = -1.0;
        int max_idx = -1;
        for (int j = 0; j < y_pred.getCols(); ++j)
        {
            if (y_pred(i, j) > max_val)
            {
                max_val = y_pred(i, j);
                max_idx = j;
            }
        }
        predictions(i, 0) = max_idx;
    }
    return predictions;
}

double calculate_accuracy(const Matrix &y_pred, const Matrix &y_true_raw)
{
    Matrix predictions = get_predictions(y_pred);
    double correct_predictions = 0;
    for (int i = 0; i < predictions.getRows(); ++i)
    {
        if (predictions(i, 0) == y_true_raw(i, 0))
        {
            correct_predictions++;
        }
    }
    return correct_predictions / predictions.getRows();
}

ConfusionMatrix::ConfusionMatrix(int num_classes)
    : m_num_classes(num_classes), m_matrix(num_classes, num_classes) {}

void ConfusionMatrix::update(const Matrix &y_pred, const Matrix &y_true_raw)
{
    Matrix predictions = get_predictions(y_pred);
    for (int i = 0; i < predictions.getRows(); ++i)
    {
        int true_label = static_cast<int>(y_true_raw(i, 0));
        int pred_label = static_cast<int>(predictions(i, 0));
        if (true_label >= 0 && true_label < m_num_classes &&
            pred_label >= 0 && pred_label < m_num_classes)
        {
            m_matrix(true_label, pred_label)++;
        }
    }
}

void ConfusionMatrix::print() const
{
    std::cout << "\n--- Confusion Matrix ---" << std::endl;
    std::cout << "Pred ->" << std::endl;
    std::cout << "True V ";
    for (int j = 0; j < m_num_classes; ++j)
    {
        std::cout << std::setw(5) << j;
    }
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int i = 0; i < m_num_classes; ++i)
    {
        std::cout << std::setw(5) << i << " |";
        for (int j = 0; j < m_num_classes; ++j)
        {
            std::cout << std::setw(5) << static_cast<int>(m_matrix(i, j));
        }
        std::cout << std::endl;
    }
}