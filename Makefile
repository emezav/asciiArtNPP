################################################################################
#
# Makefile project only supported on Linux Platforms
#
################################################################################


# Define the compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/cuda/include -Iinclude -ICommon -ICommon/UtilNPP
LDFLAGS = -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lfreeimage

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
LIB_DIR = lib

# Define source files and target executable
SRC = $(SRC_DIR)/ascii_art.cpp
TARGET = $(BIN_DIR)/asciiArtNpp.exe

# Define the default rule
all: $(TARGET)

# Build rule
build: all
	@echo "Build successful."

# Rule for building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Rule for running the application
run: $(TARGET)
	# Default filter, 80 column-width
	@echo "Default filter, 80 columns: $(DATA_DIR)/sloth_80_ascii.txt"
	./$(TARGET) $(DATA_DIR)/sloth.pgm > $(DATA_DIR)/sloth_80_ascii.txt
	@echo "Default filter, full width columns: $(DATA_DIR)/sloth_full_ascii.txt"
	./$(TARGET) $(DATA_DIR)/sloth.pgm 0 > $(DATA_DIR)/sloth_full_ascii.txt
	@echo "Kayali - x filter, full width columns: $(DATA_DIR)/sloth_full_ascii.txt"
	./$(TARGET) $(DATA_DIR)/sloth.pgm 0 6 > $(DATA_DIR)/sloth_full_kayali_x_ascii.txt
	@echo "Prewitt - x filter, full width columns: $(DATA_DIR)/sloth_full_ascii.txt"
	./$(TARGET) $(DATA_DIR)/sloth.pgm 0 8 > $(DATA_DIR)/sloth_full_prewitt_x_ascii.txt

# Clean up
clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(DATA_DIR)/*_ascii.txt

# Installation rule (not much to install, but here for completeness)
install:
	@echo "No installation required."

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Run the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make install- Install the project (if applicable)."
	@echo "  make help   - Display this help message."