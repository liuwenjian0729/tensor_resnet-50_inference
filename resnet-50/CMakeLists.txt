cmake_minimum_required(VERSION 3.13)
project(resnet-50-demo)

# CUDA & TensorRT
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
include_directories(/data/sft/TensorRT-8.6.1.6/include)
link_directories(/data/sft/TensorRT-8.6.1.6/lib)

# proto
find_package(Protobuf REQUIRED)
message(STATUS "Found Protobuf: ${PROTOBUF_VERSION}")
include_directories(${PROTOBUF_INCLUDE_DIRS})

set(PROTO_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/protos/backend_param.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE_PATH})

# Local head & source files
include_directories(
    ${PROJECT_SOURCE_DIR}
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB_RECURSE SRC_FILES 
	${PROJECT_SOURCE_DIR}/utils/*.cc
	${PROJECT_SOURCE_DIR}/tensorrt/*.cc
    ${PROJECT_SOURCE_DIR}/engine/*.cc
)

add_executable(${PROJECT_NAME} main.cc ${SRC_FILES} ${PROTO_SRCS})
target_link_libraries(${PROJECT_NAME} 
    ${CUDA_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    nvinfer
    nvparsers
    nvonnxparser
    nvinfer_plugin
)

