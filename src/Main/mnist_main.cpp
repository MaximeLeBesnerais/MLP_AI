#include "Config.hpp" // Include the new Config header
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

int mnist(const Config& config) // Modified function signature
{
    // Determine task based on config
    if (config.train) {
        std::cout << "--- MNIST Training Mode ---" << std::endl;
    } else if (config.predict) {
        std::cout << "--- MNIST Prediction Mode ---" << std::endl;
    } else {
        std::cerr << "Error: Neither train nor predict mode specified for MNIST." << std::endl;
        return 1;
    }

    // --- 1. Load and Preprocess Data ---
    std::cout << "Loading and preprocessing data..." << std::endl;
    std::string filepath = config.dataset_path.empty() ? "data/mnist_train.csv" : config.dataset_path;
    std::cout << "Loading data from: " << filepath << std::endl;
    // Consider using num_rows from config if applicable, or a portion for prediction
    auto all_data = read_csv_mnist(filepath);

    // For MNIST, we usually train on a larger portion and validate on a smaller one.
    // Prediction might use a test set or the validation set.
    // These sizes could also be part of the Config in a more advanced setup.
    int total_rows = all_data.first.getRows();
    int train_size = static_cast<int>(total_rows * 0.8); // Example: 80% for training
    if (config.predict) { // If only predicting, maybe use all data as 'validation' or a specific test set path
        train_size = 0;
    }
    int val_size = total_rows - train_size;
    if (val_size == 0 && config.train) { // Ensure there's a validation set for training
        val_size = static_cast<int>(total_rows * 0.2); // Fallback if train_size was set to total_rows
        train_size = total_rows - val_size;
    }


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
    model.add(DenseLayer(784, 128, std::make_shared<ReLU>())); // Input size 784 for MNIST
    model.add(DenseLayer(128, 10, std::make_shared<Softmax>())); // Output size 10 for 10 digits
    CategoricalCrossEntropy loss_fn;
    Adam optimizer(model.getLayers(), 0.002); // Learning rate could also be from Config

    if (config.train) {
        std::cout << "Epochs: " << config.epochs << std::endl;
        // --- 3. Early Stopping Parameters ---
        int patience = 10; // This could also be part of Config
        int epochs_no_improve = 0;
        double best_val_loss = std::numeric_limits<double>::max();
        std::vector<Matrix> best_weights;
        std::vector<Matrix> best_biases;

        std::cout << "\nStarting Training..." << std::endl;
        for (int epoch = 0; epoch < config.epochs; ++epoch) { // Use config.epochs
            if (X_train.getRows() == 0) {
                std::cerr << "Error: No training data available." << std::endl;
                return 1;
            }
            // --- Training Step on Training Data ---
            Matrix y_pred_train = model.predict(X_train);
            Matrix train_loss_grad = loss_fn.backward(y_pred_train, y_train);
            model.backward(train_loss_grad);
            optimizer.step();

            // --- Validation Step on Validation Data ---
            if (X_val.getRows() > 0) {
                Matrix y_pred_val = model.predict(X_val);
                double val_loss = loss_fn.calculate(y_pred_val, y_val);

                if (epoch % 5 == 0) {
                    std::cout << "Epoch: " << epoch << ", Validation Loss: " << val_loss << std::endl;
                }

                // --- Early Stopping Logic ---
                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    epochs_no_improve = 0;
                    // Save a snapshot of the best model weights
                    best_weights.clear();
                    best_biases.clear();
                    for (auto &layer : model.getLayers()) {
                        best_weights.push_back(layer.getWeights());
                        best_biases.push_back(layer.getBiases());
                    }
                } else {
                    epochs_no_improve++;
                }

                if (epochs_no_improve >= patience) {
                    std::cout << "\nEarly stopping triggered at epoch " << epoch << "!" << std::endl;
                    if (!best_weights.empty()) {
                        for (size_t i = 0; i < model.getLayers().size(); ++i) {
                            model.getLayers()[i].setWeights(best_weights[i]);
                            model.getLayers()[i].setBiases(best_biases[i]);
                        }
                    }
                    break;
                }
            } else {
                 if (epoch % 5 == 0) { // Still print training loss if no validation
                    double train_loss_val = loss_fn.calculate(y_pred_train, y_train);
                    std::cout << "Epoch: " << epoch << ", Training Loss: " << train_loss_val << std::endl;
                }
            }
        }
        std::cout << "\nTraining Complete." << std::endl;

        if (!config.save_model_path.empty()) {
            // TODO: Save model to config.save_model_path (consider saving best model from early stopping)
            std::cout << "Placeholder: Would save best model to " << config.save_model_path << std::endl;
        }

        // --- Final Evaluation after Training ---
        if (X_val.getRows() > 0) {
            std::cout << "\n--- Evaluation using Best Model on Validation Set ---" << std::endl;
            Matrix final_preds_val = model.predict(X_val);
            double accuracy_val = calculate_accuracy(final_preds_val, y_val_raw);
            std::cout << "Validation Accuracy: " << accuracy_val * 100.0 << "%" << std::endl;
        } else {
            std::cout << "\n--- Evaluation using Final Model on Training Set (No Validation Set) ---" << std::endl;
            Matrix final_preds_train = model.predict(X_train);
            double accuracy_train = calculate_accuracy(final_preds_train, y_train_raw); // Assuming y_train_raw is available
            std::cout << "Training Accuracy: " << accuracy_train * 100.0 << "%" << std::endl;
        }


    } else if (config.predict) {
        if (config.load_model_path.empty()) {
            std::cerr << "Error: Model path must be provided for prediction." << std::endl;
            return 1;
        }
        // TODO: Load model from config.load_model_path
        std::cout << "Placeholder: Would load model from " << config.load_model_path << std::endl;
        std::cout << "Note: Prediction logic assumes model is loaded and ready." << std::endl;

        // Perform prediction (using validation set as an example, or X_all if appropriate)
        Matrix X_predict = X_val.getRows() > 0 ? X_val : X_all; // Use validation set if available, else all data
        Matrix y_predict_raw = y_val_raw.getRows() > 0 ? y_val_raw : all_data.second; // Corresponding labels

        if (X_predict.getRows() == 0) {
            std::cerr << "Error: No data available for prediction." << std::endl;
            return 1;
        }

        normalize_features(X_predict); // Ensure prediction data is also normalized

        std::cout << "\n--- Prediction Results ---" << std::endl;
        Matrix predictions = model.predict(X_predict);
        double accuracy = calculate_accuracy(predictions, y_predict_raw); // y_predict_raw needs to be the non-one-hot-encoded version
        std::cout << "Prediction Accuracy on " << (X_val.getRows() > 0 ? "validation" : "full") << " set: " << accuracy * 100.0 << "%" << std::endl;

        // Print some sample predictions (e.g., first 5)
        std::cout << "\nSample Predictions (Predicted vs True):" << std::endl;
        for (int i = 0; i < std::min(5, predictions.getRows()); ++i) {
            int pred_label = predictions.argmax(i);
            int true_label = static_cast<int>(y_predict_raw(i,0));
            std::cout << "Sample " << i << ": Predicted=" << pred_label << ", True=" << true_label << std::endl;
        }
    }

    return 0;
}
