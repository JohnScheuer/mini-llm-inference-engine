cmake_minimum_required(VERSION 3.10)
project(MiniLLM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_executable(llm ${SOURCES})

target_include_directories(llm PUBLIC src)

if(OpenMP_CXX_FOUND)
    target_link_libraries(llm PRIVATE OpenMP::OpenMP_CXX)
endif()