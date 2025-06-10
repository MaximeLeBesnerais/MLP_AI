#include <iostream>
#include "Model.hpp"
#include "layers/DenseLayer.hpp"
#include "activations/ReLU.hpp"
#include "activations/LinearActivation.hpp" // Our new linear activation
#include "losses/MeanSquaredError.hpp" // Use MSE for regression
#include "optimizers/Adam.hpp"
#include "utils/DataHandler.hpp"

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

int boston() {
    std::cout << "--- Boston Housing Regression Task ---" << std::endl;

    // --- 1. Load and Preprocess Data ---
    auto raw_data_with_labels = read_csv_boston("data/boston_housing.csv");
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

    // --- 3. Train the Model ---
    MeanSquaredError loss_fn;
    Adam optimizer(model.getLayers(), 0.01);
    int epochs = 100;

    std::cout << "\nStarting Training..." << std::endl;
    for (int epoch = 0; epoch <= epochs; ++epoch) {
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
    Matrix final_preds = model.predict(X_val);
    std::cout << "Final Validation MSE: " << loss_fn.calculate(final_preds, y_val) << std::endl;
    std::cout << "\nSample Predictions vs True Values:" << std::endl;
    for(int i=0; i<10; ++i) {
        std::cout << "Pred: " << final_preds(i,0) << ", True: " << y_val(i,0) << std::endl;
    }
    
    return 0;
}