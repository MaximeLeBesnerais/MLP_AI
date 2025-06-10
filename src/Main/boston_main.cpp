#include "Config.hpp" // Include the new Config header
#include <iostream>
#include "Model.hpp"
#include "DenseLayer.hpp"
#include "ReLU.hpp"
#include "LinearActivation.hpp" // Our new linear activation
#include "MeanSquaredError.hpp" // Use MSE for regression
#include "Adam.hpp"
#include "DataHandler.hpp"

// Helper to separate last column as target
std::pair<Matrix, Matrix> separate_features_target(const Matrix& data) {
    int feature_cols = data.getCols() - 1;
    Matrix features(data.getRows(), feature_cols);
    Matrix target(data.getRows(), 1);

    for (int i = 0; i < data.getRows(); ++i) {
        for (int j = 0; j < feature_cols; ++j) {
            features(i, j) = data(i, j);
        }
        target(i, 0) = data(i, feature_cols);
    }
    return {features, target};
}

int boston(const Config& config) { // Modified function signature
    std::cout << "--- Boston Housing Regression Task ---" << std::endl;

    // --- 1. Load and Preprocess Data ---
    std::string filepath = config.dataset_path.empty() ? "data/boston_housing.csv" : config.dataset_path;
    std::cout << "Loading data from: " << filepath << std::endl;
    auto raw_data_with_labels = read_csv_boston(filepath);
    auto separated_data = separate_features_target(raw_data_with_labels);
    Matrix X_all = separated_data.first;
    Matrix y_all = separated_data.second;

    // Split data
    int train_size = 400;
    Matrix X_train = X_all.slice(0, train_size);
    Matrix y_train = y_all.slice(0, train_size);
    Matrix X_val = X_all.slice(train_size, X_all.getRows());
    Matrix y_val = y_all.slice(train_size, y_all.getRows());

    // Scale features
    StandardScaler scaler;
    X_train = scaler.fit_transform(X_train);
    X_val = scaler.transform(X_val);

    // --- 2. Define Regression Model ---
    Model model;
    model.add(DenseLayer(X_train.getCols(), 64, std::make_shared<ReLU>()));
    model.add(DenseLayer(64, 64, std::make_shared<ReLU>()));
    model.add(DenseLayer(64, 1, std::make_shared<LinearActivation>())); // Output layer: 1 neuron, linear activation

    // --- 3. Define Model Components ---
    MeanSquaredError loss_fn;
    Adam optimizer(model.getLayers(), 0.01); // Learning rate can also be part of Config

    if (config.train) {
        std::cout << "\n--- Training Mode ---" << std::endl;
        std::cout << "Epochs: " << config.epochs << std::endl;

        std::cout << "\nStarting Training..." << std::endl;
        for (int epoch = 0; epoch <= config.epochs; ++epoch) { // Use config.epochs
            Matrix y_pred = model.predict(X_train);
            Matrix grad = loss_fn.backward(y_pred, y_train);
            model.backward(grad);
            optimizer.step();

            if (epoch % 10 == 0) {
                Matrix val_pred = model.predict(X_val);
                double val_loss = loss_fn.calculate(val_pred, y_val);
                std::cout << "Epoch: " << epoch << ", Validation MSE: " << val_loss << std::endl;
            }
        }
        std::cout << "\nTraining Complete." << std::endl;
        Matrix final_train_preds = model.predict(X_val); // Predict on validation set after training
        std::cout << "Final Validation MSE after training: " << loss_fn.calculate(final_train_preds, y_val) << std::endl;

        if (!config.save_model_path.empty()) {
            // TODO: Save model to config.save_model_path
            std::cout << "Placeholder: Would save model to " << config.save_model_path << std::endl;
        }

    } else if (config.predict) {
        std::cout << "\n--- Prediction Mode ---" << std::endl;
        if (config.load_model_path.empty()) {
            std::cerr << "Error: Model path must be provided for prediction." << std::endl;
            return 1;
        }
        // TODO: Load model from config.load_model_path
        std::cout << "Placeholder: Would load model from " << config.load_model_path << std::endl;
        std::cout << "Note: Prediction logic assumes model is loaded and ready." << std::endl;

        // Perform prediction (using validation set as an example)
        Matrix predictions = model.predict(X_val); // Or X_all if no separate validation set for pure prediction
        std::cout << "Validation MSE using (potentially un-trained/default) model: " << loss_fn.calculate(predictions, y_val) << std::endl;
        std::cout << "\nSample Predictions vs True Values (on validation set):" << std::endl;
        int num_samples_to_show = std::min(10, predictions.getRows()); // Show up to 10 samples
        for(int i=0; i < num_samples_to_show; ++i) {
            std::cout << "Pred: " << predictions(i,0) << ", True: " << y_val(i,0) << std::endl;
        }

    } else {
        std::cerr << "Error: Neither train nor predict mode specified." << std::endl;
        return 1;
    }
    
    return 0;
}