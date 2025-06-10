#include "Mains.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <task>" << std::endl;
        std::cerr << "Available tasks: boston, mnist" << std::endl;
        return 1;
    }

    std::string task = argv[1];
    if (task == "boston")
    {
        return boston();
    }
    else if (task == "mnist")
    {
        return mnist();
    }
    else
    {
        std::cerr << "Unknown task: " << task << std::endl;
        return 1;
    }
}
