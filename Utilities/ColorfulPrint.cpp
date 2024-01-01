#include "ColorfulPrint.h"

namespace Utilities
{
    const std::string GreenHead() {
        return "\x1b[1;30;92m";
    }

    const std::string RedHead() {
        return "\x1b[1;30;91m";
    }

    const std::string YellowHead() {
        return "\x1b[1;30;93m";
    }

    const std::string CyanHead() {
        return "\x1b[1;30;96m";
    }

    const std::string GreenTail() {
        return "\x1b[0m";
    }

    const std::string RedTail() {
        return "\x1b[0m";
    }

    const std::string YellowTail() {
        return "\x1b[0m";
    }

    const std::string CyanTail() {
        return "\x1b[0m";
    }

    void PrintError(const std::string& message, const int return_code) {
#if PRINT_LEVEL >= PRINT_ERROR
        std::cerr << RedHead() << message << RedTail() << std::endl;
        throw return_code;
#endif
    }

    void PrintWarning(const std::string& message) {
#if PRINT_LEVEL >= PRINT_ERROR_AND_WARNING
        std::cout << YellowHead() << message << YellowTail() << std::endl;
#endif
    }

    void PrintInfo(const std::string& message) {
#if PRINT_LEVEL >= PRINT_ALL
        std::cout << CyanHead() << message << CyanTail() << std::endl;
#endif
    }

    void PrintSuccess(const std::string& message) {
        std::cout << GreenHead() << message << GreenTail() << std::endl;
    }
}
