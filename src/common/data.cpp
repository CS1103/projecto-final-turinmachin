#include "common/data.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include "config.h"

namespace common {

    auto get_data_path() -> std::string {
        std::string installed_path = std::string(SHARE_DIR);
        std::string dev_path = "../share";

        if (std::filesystem::exists(installed_path)) {
            return installed_path;
        }

        return dev_path;
    }

    auto get_data_file_path(const std::string& file_path) -> std::string {
        const std::string full_path = get_data_path() + "/" + file_path;
        if (!std::filesystem::exists(full_path)) {
            throw std::runtime_error("Data file does not exist: " + file_path);
        }

        return full_path;
    }

    auto get_data_file(const std::string& file_path) -> std::ifstream {
        const std::string full_path = get_data_file_path(file_path);
        return std::ifstream(full_path);
    }

}  // namespace common
