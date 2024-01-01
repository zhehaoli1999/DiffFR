#ifndef  __COLORFUL_PRINT_H__
#define __COLORFUL_PRINT_H__

#include <string>
#include <iostream>

#define PRINT_NOTHING           0
#define PRINT_ERROR             1
#define PRINT_ERROR_AND_WARNING 2
#define PRINT_ALL               3
#define PRINT_LEVEL             PRINT_ALL

namespace Utilities
{
    // Colorful print.
    const std::string GreenHead();
    const std::string RedHead();
    const std::string YellowHead();
    const std::string CyanHead();
    const std::string GreenTail();
    const std::string RedTail();
    const std::string YellowTail();
    const std::string CyanTail();
    // Use return_code = -1 unless you want to customize it.
    void PrintError(const std::string& message, const int return_code = -1);
    void PrintWarning(const std::string& message);
    void PrintInfo(const std::string& message);
    void PrintSuccess(const std::string& message);
}
#endif // !
