cmake_minimum_required(VERSION 3.10)
project(TorchScript_Cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Specify the path to LibTorch
set(Torch_DIR "your_path")
set(OpenCV_DIR "your_path")

# Find necessary packages
find_package(Torch REQUIRED)  
find_package(OpenCV REQUIRED)

# Define the executable
add_executable(main src/main.cpp src/params.h)

# Link against Torch and OpenCV
target_link_libraries(main "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Set C++ standard property
set_property(TARGET main PROPERTY CXX_STANDARD 17)
