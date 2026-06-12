cmake_minimum_required(VERSION 3.10)
project(MiniLLM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Flags de otimização agressiva
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")

find_package(OpenMP REQUIRED)

# Pega TODOS os .cpp (incluindo subpastas)
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(llm ${SOURCES})

# Permite incluir headers de src/ e src/benchmark/
target_include_directories(llm PUBLIC src src/benchmark)

if(OpenMP_CXX_FOUND)
    target_link_libraries(llm PRIVATE OpenMP::OpenMP_CXX)
endif()