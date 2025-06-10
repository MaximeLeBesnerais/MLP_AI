#include <iostream>
#include <limits>
#include <vector>

#include "Model.hpp"
#include "DenseLayer.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "CategoricalCrossEntropy.hpp"
#include "Adam.hpp"
#include "DataHandler.hpp"
#include "Evaluation.hpp"
#include "ElasticNetRegularizer.hpp"

int mnist()
{
    std::cout << "--- Training with Early Stopping ---" << std::endl;

    // --- 1. Load and Preprocess Data ---
    std::cout << "Loading and preprocessing data..." << std::endl;
    auto all_data = read_csv_mnist("data/mnist_train.csv");
    int train_size = 5000;
    int val_size = 1000;
    Matrix X_train = all_data.first.slice(0, train_size);
    Matrix y_train_raw = all_data.second.slice(0, train_size);
    Matrix X_val = all_data.first.slice(train_size, train_size + val_size);
    Matrix y_val_raw = all_data.second.slice(train_size, train_size + val_size);
    normalize_features(X_train);
    normalize_features(X_val);
    Matrix y_train = one_hot_encode(y_train_raw, 10);
    Matrix y_val = one_hot_encode(y_val_raw, 10);

    // --- 2. Define Model and Training Parameters ---
    Model model;
    model.add(DenseLayer(784, 128, std::make_shared<ReLU>()));
    model.add(DenseLayer(128, 10, std::make_shared<Softmax>()));
    CategoricalCrossEntropy loss_fn;
    Adam optimizer(model.getLayers(), 0.002);
    int max_epochs = 200; // The maximum we're willing to train for

    // --- 3. Early Stopping Parameters ---
    int patience = 10;
    int epochs_no_improve = 0;
    double best_val_loss = std::numeric_limits<double>::max();
    std::vector<Matrix> best_weights;
    std::vector<Matrix> best_biases;

    std::cout << "\nStarting Training..." << std::endl;
    for (int epoch = 0; epoch < max_epochs; ++epoch)
    {
        // --- Training Step on Training Data ---
        Matrix y_pred_train = model.predict(X_train);
        Matrix train_loss_grad = loss_fn.backward(y_pred_train, y_train);
        model.backward(train_loss_grad);
        optimizer.step();

        // --- Validation Step on Validation Data ---
        Matrix y_pred_val = model.predict(X_val);
        double val_loss = loss_fn.calculate(y_pred_val, y_val);

        if (epoch % 5 == 0)
        {
            std::cout << "Epoch: " << epoch << ", Validation Loss: " << val_loss << std::endl;
        }

        // --- Early Stopping Logic ---
        if (val_loss < best_val_loss)
        {
            best_val_loss = val_loss;
            epochs_no_improve = 0;
            // Save a snapshot of the best model weights
            best_weights.clear();
            best_biases.clear();
            for (auto &layer : model.getLayers())
            {
                best_weights.push_back(layer.getWeights());
                best_biases.push_back(layer.getBiases());
            }
        }
        else
        {
            epochs_no_improve++;
        }

        if (epochs_no_improve >= patience)
        {
            std::cout << "\nEarly stopping triggered at epoch " << epoch << "!" << std::endl;
            // Restore the best weights found
            if (!best_weights.empty())
            {
                for (size_t i = 0; i < model.getLayers().size(); ++i)
                {
                    model.getLayers()[i].setWeights(best_weights[i]);
                    model.getLayers()[i].setBiases(best_biases[i]);
                }
            }
            break; // Exit the training loop
        }
    }

    // --- 4. Final Evaluation using the Best Model ---
    std::cout << "\n--- Evaluation using Best Model ---" << std::endl;
    Matrix final_preds = model.predict(X_val);
    double accuracy = calculate_accuracy(final_preds, y_val_raw);
    std::cout << "Validation Accuracy: " << accuracy * 100.0 << "%" << std::endl;

    return 0;
}
