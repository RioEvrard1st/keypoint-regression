cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

project(keypoint-regression)

# onnx
# find_package(ONNX REQUIRED)
# include_directories(${ONNX_INCLUDE_DIRS})
# message(STATUS "ONNX library status:")
# message(STATUS "    version: ${ONNXVERSION}")
# message(STATUS "    libraries: ${ONNX_LIBS}")

# libtorch
# set(CMAKE_PREFIX_PATH /home/libtorch)
# find_package(Torch REQUIRED)
# include_directories(${TORCH_INCLUDE_DIRS})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# message(STATUS "Pytorch status:")
# message(STATUS "    version: ${TORCH_VERSION}")
# message(STATUS "    libraries: ${TORCH_LIBRARIES}")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
