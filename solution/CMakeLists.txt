cmake_minimum_required(VERSION 3.14)
project(mlsys_trackA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_executable(mlsys main.cpp scheduler.cpp)
target_link_libraries(mlsys PRIVATE nlohmann_json::nlohmann_json)
