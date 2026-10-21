cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(tch)

find_package(Torch REQUIRED)

add_library(tch STATIC torch_api.cpp torch_api_generated.cpp)
target_link_libraries(tch "${TORCH_LIBRARIES}")
set_property(TARGET tch PROPERTY CXX_STANDARD 11)
install(TARGETS tch DESTINATION .)
