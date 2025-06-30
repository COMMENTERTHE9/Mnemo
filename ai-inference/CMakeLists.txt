cmake_minimum_required(VERSION 3.18)
project(video_memory_ai_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find gRPC
find_package(Protobuf CONFIG QUIET)
find_package(gRPC CONFIG QUIET)

if(NOT gRPC_FOUND)
    message(WARNING "gRPC not found, building without gRPC support")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/relevance_ai.cpp
    src/narrative_ai.cpp
    src/reconciler.cpp
)

# Create executable
add_executable(ai_inference ${SOURCES})

# Link libraries
if(gRPC_FOUND)
    target_link_libraries(ai_inference 
        "${TORCH_LIBRARIES}"
        gRPC::grpc++
        protobuf::libprotobuf
    )
else()
    target_link_libraries(ai_inference 
        "${TORCH_LIBRARIES}"
    )
endif()

# Set C++ properties
set_property(TARGET ai_inference PROPERTY CXX_STANDARD 17)

# Models will be copied during Docker runtime, not build time