#include "Mains.hpp"
#include "CmdParser.hpp" // Our new parser
#include "Config.hpp"    // Our new Config header
#include <iostream>
#include <string>

// This function will eventually replace your old boston() and mnist() functions
void run_task(const Config &config)
{
    if (config.task_mode == "boston")
    {
        boston(config);
    }
    else if (config.task_mode == "mnist")
    {
        mnist(config);
    }
    else
    {
        std::cerr << "Error: Unknown task mode: " << config.task_mode << std::endl;
        // Potentially return an error code or throw an exception
    }
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