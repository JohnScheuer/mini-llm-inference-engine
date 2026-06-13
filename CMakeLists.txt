cmake_minimum_required(VERSION 3.15)
project(FlashLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops")

find_package(OpenMP REQUIRED)

file(GLOB SOURCES "src/*.cpp")
add_executable(llm ${SOURCES})

target_link_libraries(llm PUBLIC OpenMP::OpenMP_CXX)
