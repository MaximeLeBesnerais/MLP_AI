#include "Mains.hpp"
#include "CmdParser.hpp" // Our new parser
#include <iostream>
#include <string>

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

// This function will eventually replace your old boston() and mnist() functions
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

    std::cout << "\nTODO: Implement the training/prediction logic here." << std::endl;

    // --- We will add the logic back in the next step ---
}

int main(int argc, char *argv[])
{
    CmdParser parser(argc, argv);
    Config config;

    // --- Parse all arguments ---
    const std::string &task = parser.get_option("--mode");
    if (task.empty())
    {
        std::cerr << "Error: Task mode is required. Use --mode <mnist|boston>" << std::endl;
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