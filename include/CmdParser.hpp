#ifndef CMD_PARSER_HPP
#define CMD_PARSER_HPP

#include <string>
#include <vector>
#include <map>
#include <algorithm>

class CmdParser
{
public:
    CmdParser(int &argc, char **argv)
    {
        for (int i = 1; i < argc; ++i)
        {
            m_tokens.push_back(std::string(argv[i]));
        }
    }

    // Get the value of an option, e.g., for "--mode mnist", get_option("--mode") returns "mnist"
    const std::string &get_option(const std::string &option) const
    {
        auto it = std::find(m_tokens.begin(), m_tokens.end(), option);
        if (it != m_tokens.end() && ++it != m_tokens.end())
        {
            return *it;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    // Check if an option exists, e.g., for "--train"
    bool option_exists(const std::string &option) const
    {
        return std::find(m_tokens.begin(), m_tokens.end(), option) != m_tokens.end();
    }

private:
    std::vector<std::string> m_tokens;
};

#endif // CMD_PARSER_HPP