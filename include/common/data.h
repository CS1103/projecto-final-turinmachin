#ifndef INCLUDE_COMMON_DATA_H
#define INCLUDE_COMMON_DATA_H

#include <fstream>
#include <string>

namespace common {

    auto get_data_path() -> std::string;

    auto get_data_file_path(const std::string& file_path) -> std::string;

    auto get_data_file(const std::string& path) -> std::ifstream;

}  // namespace common

#endif
