#include "Mains.hpp"
#include "utils/CmdParser.hpp" // Our new parser
#include <iostream>
#include <string>

// Include all necessary headers for the actual implementations
#include "Model.hpp"
#include "layers/DenseLayer.hpp"
#include "activations/ReLU.hpp"
#include "activations/LinearActivation.hpp"
#include "activations/Softmax.hpp"
#include "losses/MeanSquaredError.hpp"
#include "losses/CategoricalCrossEntropy.hpp"
#include "optimizers/Adam.hpp"
#include "utils/DataHandler.hpp"
#include "utils/Evaluation.hpp"
#include <limits>
#include <vector>

// A struct to hold our configuration
struct Config
{
    std::string task_mode; // "mnist" or "boston"
    std::string dataset_path;
    std::string load_model_path;
    std::string save_model_path;
    int epochs = 100; // Default value
    bool train = false;
    bool predict = false;
};

// Helper to separate last column as target (for Boston dataset)
static std::pair<Matrix, Matrix> separate_features_target(const Matrix& data) {
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

// Forward declarations for the specific task implementations
void run_boston_task(const Config &config);
void run_mnist_task(const Config &config);

// This function dispatches to the appropriate task based on configuration
void run_task(const Config &config)
{
    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "Task Mode: " << config.task_mode << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Training Enabled: " << (config.train ? "Yes" : "No") << std::endl;
    std::cout << "Prediction Enabled: " << (config.predict ? "Yes" : "No") << std::endl;
    if (!config.dataset_path.empty())
        std::cout << "Dataset Path: " << config.dataset_path << std::endl;
    if (!config.load_model_path.empty())
        std::cout << "Load Model Path: " << config.load_model_path << std::endl;
    if (!config.save_model_path.empty())
        std::cout << "Save Model Path: " << config.save_model_path << std::endl;

    // Dispatch to the appropriate task
    if (config.task_mode == "boston")
    {
        run_boston_task(config);
    }
    else if (config.task_mode == "mnist")
    {
        run_mnist_task(config);
    }
    else
    {
        std::cerr << "Error: Unknown task mode '" << config.task_mode << "'. Use 'boston' or 'mnist'." << std::endl;
    }
}

void run_boston_task(const Config &config)
{
    std::cout << "\n--- Boston Housing Regression Task ---" << std::endl;

    // Determine dataset path
    std::string dataset_path = config.dataset_path.empty() ? "data/boston_housing.csv" : config.dataset_path;

    if (config.train)
    {
        std::cout << "=== TRAINING MODE ===" << std::endl;
        
        // --- 1. Load and Preprocess Data ---
        auto raw_data_with_labels = read_csv_boston(dataset_path);
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

        // Load existing model if specified
        if (!config.load_model_path.empty())
        {
            std::cout << "Loading existing model from: " << config.load_model_path << std::endl;
            try {
                model.load(config.load_model_path);
                std::cout << "Model loaded successfully!" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not load model: " << e.what() << std::endl;
                std::cout << "Continuing with fresh model..." << std::endl;
            }
        }

        // --- 3. Train the Model ---
        MeanSquaredError loss_fn;
        Adam optimizer(model.getLayers(), 0.01);

        std::cout << "\nStarting Training for " << config.epochs << " epochs..." << std::endl;
        for (int epoch = 0; epoch <= config.epochs; ++epoch) {
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
        
        // Save model if specified
        if (!config.save_model_path.empty())
        {
            std::cout << "Saving model to: " << config.save_model_path << std::endl;
            try {
                model.save(config.save_model_path);
                std::cout << "Model saved successfully!" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error saving model: " << e.what() << std::endl;
            }
        }

        std::cout << "\nSample Predictions vs True Values:" << std::endl;
        for(int i=0; i<10 && i<final_preds.getRows(); ++i) {
            std::cout << "Pred: " << final_preds(i,0) << ", True: " << y_val(i,0) << std::endl;
        }
    }
    else if (config.predict)
    {
        std::cout << "=== PREDICTION MODE ===" << std::endl;
        
        // Load data for prediction
        auto raw_data_with_labels = read_csv_boston(dataset_path);
        auto separated_data = separate_features_target(raw_data_with_labels);
        Matrix X_all = separated_data.first;
        Matrix y_all = separated_data.second; // For comparison if available

        // Scale features (Note: In production, you'd want to save/load scaler parameters)
        StandardScaler scaler;
        Matrix X_scaled = scaler.fit_transform(X_all);

        // --- Create and Load Model ---
        Model model;
        model.add(DenseLayer(X_scaled.getCols(), 64, std::make_shared<ReLU>()));
        model.add(DenseLayer(64, 64, std::make_shared<ReLU>()));
        model.add(DenseLayer(64, 1, std::make_shared<LinearActivation>()));

        std::cout << "Loading model from: " << config.load_model_path << std::endl;
        try {
            model.load(config.load_model_path);
            std::cout << "Model loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return;
        }

        // --- Make Predictions ---
        Matrix predictions = model.predict(X_scaled);
        
        std::cout << "\nPredictions:" << std::endl;
        for(int i = 0; i < std::min(20, predictions.getRows()); ++i) {
            std::cout << "Sample " << i+1 << " - Predicted: " << predictions(i,0);
            if (i < y_all.getRows()) {
                std::cout << ", Actual: " << y_all(i,0);
            }
            std::cout << std::endl;
        }
        
        if (predictions.getRows() > 20) {
            std::cout << "... and " << (predictions.getRows() - 20) << " more predictions." << std::endl;
        }
    }
}

void run_mnist_task(const Config &config)
{
    std::cout << "\n--- MNIST Classification Task ---" << std::endl;

    // Determine dataset paths
    std::string train_dataset_path = config.dataset_path.empty() ? "data/mnist_train.csv" : config.dataset_path;
    std::string test_dataset_path = "data/mnist_test.csv"; // Could be made configurable

    if (config.train)
    {
        std::cout << "=== TRAINING MODE ===" << std::endl;
        
        // --- 1. Load and Preprocess Data ---
        std::cout << "Loading and preprocessing data..." << std::endl;
        auto all_data = read_csv_mnist(train_dataset_path);
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
        
        // Load existing model if specified
        if (!config.load_model_path.empty())
        {
            std::cout << "Loading existing model from: " << config.load_model_path << std::endl;
            try {
                model.load(config.load_model_path);
                std::cout << "Model loaded successfully!" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not load model: " << e.what() << std::endl;
                std::cout << "Continuing with fresh model..." << std::endl;
            }
        }
        
        CategoricalCrossEntropy loss_fn;
        Adam optimizer(model.getLayers(), 0.002);

        // --- 3. Early Stopping Parameters ---
        int patience = 10;
        int epochs_no_improve = 0;
        double best_val_loss = std::numeric_limits<double>::max();
        std::vector<Matrix> best_weights;
        std::vector<Matrix> best_biases;

        std::cout << "\nStarting Training for up to " << config.epochs << " epochs..." << std::endl;
        for (int epoch = 0; epoch < config.epochs; ++epoch)
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
                double accuracy = calculate_accuracy(y_pred_val, y_val_raw);
                std::cout << "Epoch: " << epoch << ", Validation Loss: " << val_loss 
                         << ", Accuracy: " << accuracy * 100.0 << "%" << std::endl;
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
        std::cout << "Final Validation Accuracy: " << accuracy * 100.0 << "%" << std::endl;

        // Save model if specified
        if (!config.save_model_path.empty())
        {
            std::cout << "Saving model to: " << config.save_model_path << std::endl;
            try {
                model.save(config.save_model_path);
                std::cout << "Model saved successfully!" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error saving model: " << e.what() << std::endl;
            }
        }
    }
    else if (config.predict)
    {
        std::cout << "=== PREDICTION MODE ===" << std::endl;
        
        // --- Load Data for Prediction ---
        std::cout << "Loading data for prediction..." << std::endl;
        auto test_data = read_csv_mnist(test_dataset_path);
        Matrix X_test = test_data.first;
        Matrix y_test_raw = test_data.second; // For comparison if available
        normalize_features(X_test);

        // --- Create and Load Model ---
        Model model;
        model.add(DenseLayer(784, 128, std::make_shared<ReLU>()));
        model.add(DenseLayer(128, 10, std::make_shared<Softmax>()));

        std::cout << "Loading model from: " << config.load_model_path << std::endl;
        try {
            model.load(config.load_model_path);
            std::cout << "Model loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return;
        }

        // --- Make Predictions ---
        Matrix predictions = model.predict(X_test);
        
        // Convert predictions to class labels
        std::cout << "\nPredictions:" << std::endl;
        for(int i = 0; i < std::min(20, predictions.getRows()); ++i) {
            // Find the class with highest probability
            int predicted_class = 0;
            double max_prob = predictions(i, 0);
            for(int j = 1; j < predictions.getCols(); ++j) {
                if (predictions(i, j) > max_prob) {
                    max_prob = predictions(i, j);
                    predicted_class = j;
                }
            }
            
            std::cout << "Sample " << i+1 << " - Predicted: " << predicted_class;
            if (i < y_test_raw.getRows()) {
                std::cout << ", Actual: " << (int)y_test_raw(i,0);
            }
            std::cout << " (confidence: " << max_prob << ")" << std::endl;
        }
        
        if (predictions.getRows() > 20) {
            std::cout << "... and " << (predictions.getRows() - 20) << " more predictions." << std::endl;
        }

        // Calculate overall accuracy if we have labels
        if (y_test_raw.getRows() > 0) {
            double accuracy = calculate_accuracy(predictions, y_test_raw);
            std::cout << "\nOverall Test Accuracy: " << accuracy * 100.0 << "%" << std::endl;
        }
    }
}

void print_usage() {
    std::cout << "\nUsage: ./mlp --mode <mnist|boston> <--train|--predict> [options]\n" << std::endl;
    std::cout << "Required arguments:" << std::endl;
    std::cout << "  --mode <task>          Task mode: 'mnist' or 'boston'" << std::endl;
    std::cout << "  --train OR --predict   Training or prediction mode" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --epochs <num>         Number of training epochs (default: 100)" << std::endl;
    std::cout << "  --dataset <path>       Path to dataset file" << std::endl;
    std::cout << "  --load <path>          Load existing model from file" << std::endl;
    std::cout << "  --save <path>          Save trained model to file" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./mlp --mode mnist --train --epochs 150 --save models/mnist_model.txt" << std::endl;
    std::cout << "  ./mlp --mode mnist --predict --load models/mnist_model.txt" << std::endl;
    std::cout << "  ./mlp --mode boston --train --dataset data/custom_boston.csv --save models/boston_model.txt" << std::endl;
    std::cout << "  ./mlp --mode boston --predict --load models/boston_model.txt" << std::endl;
}

int main(int argc, char *argv[])
{
    // Show help if no arguments or help requested
    if (argc == 1) {
        std::cout << "Error: No arguments provided." << std::endl;
        print_usage();
        return 1;
    }

    CmdParser parser(argc, argv);
    
    // Check for help
    if (parser.option_exists("--help") || parser.option_exists("-h")) {
        print_usage();
        return 0;
    }

    Config config;

    // --- Parse all arguments ---
    const std::string &task = parser.get_option("--mode");
    if (task.empty())
    {
        std::cerr << "Error: Task mode is required. Use --mode <mnist|boston>" << std::endl;
        print_usage();
        return 1;
    }
    config.task_mode = task;

    config.train = parser.option_exists("--train");
    config.predict = parser.option_exists("--predict");

    const std::string &epochs_str = parser.get_option("--epochs");
    if (!epochs_str.empty())
    {
        config.epochs = std::stoi(epochs_str);
    }

    config.dataset_path = parser.get_option("--dataset");
    config.load_model_path = parser.get_option("--load");
    config.save_model_path = parser.get_option("--save");

    // --- Basic validation ---
    if (config.train == config.predict)
    {
        std::cerr << "Error: Please specify exactly one of --train or --predict." << std::endl;
        return 1;
    }
    if (config.predict && config.load_model_path.empty())
    {
        std::cerr << "Error: Prediction mode requires a model file. Use --load <path_to_model>" << std::endl;
        return 1;
    }

    run_task(config);

    return 0;
}