CXX = g++
# Flags de otimização extremas
CXXFLAGS = -std=c++17 -O3 -march=native -mavx2 -mfma -Wall -Wextra -I./src

# Pega todos os arquivos .cpp da pasta src
SRCS = $(wildcard src/*.cpp)
OBJS = $(SRCS:.cpp=.o)

TARGET = mini_llm

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build completo! Executável gerado: $(TARGET)"

# Regra para compilar os objetos
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)