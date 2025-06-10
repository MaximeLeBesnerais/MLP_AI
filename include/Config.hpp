#ifndef CONFIG_HPP
#define CONFIG_HPP

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

#endif // CONFIG_HPP
