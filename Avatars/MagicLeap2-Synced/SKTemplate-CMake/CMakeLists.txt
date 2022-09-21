# To use this template, 
cmake_minimum_required(VERSION 3.11)
project(SKNativeTemplate VERSION "0.1.0" LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# set(CMAKE_CXX_FLAGS -rdynamic) #backtrace improvement?

add_definitions(-w) # disable compiler warnings

link_directories(${CMAKE_SOURCE_DIR}/../StereoKit/)
link_directories(${CMAKE_SOURCE_DIR}/../libraries/)
link_directories(${CMAKE_SOURCE_DIR}/../mediapipe-master/build)
include_directories(
  ${CMAKE_SOURCE_DIR}/../StereoKit/StereoKitC
  ${CMAKE_SOURCE_DIR}/../mlsdk/include
  ${CMAKE_SOURCE_DIR}/../mlsdk/include
  ${CMAKE_SOURCE_DIR}/../libraries/pytorch_headers
  ${CMAKE_SOURCE_DIR}/../libraries/tensorflowlite_headers
  ${CMAKE_SOURCE_DIR}/../libraries/tensorflowlite_gpu_headers
)

# Make list of libraries to link against
list(APPEND SI_LINK_LIBS 
  StereoKitC 
  log 
  camera.magicleap 
  perception.magicleap
  libpytorch_jni.so
  libfbjni.so
  libtensorflowlite_jni.so 
  libtensorflowlite_gpu_jni.so 
)

# Add source files to our target executable
add_executable( SKNativeTemplate
  src/main.cpp
)
add_library(PoseModel SHARED
  src/PoseModel.cpp
)
target_precompile_headers(SKNativeTemplate PRIVATE ../libraries/pytorch_headers/torch/script.h) # Crucial, otherwise takes forever to compile
target_precompile_headers(PoseModel PRIVATE ../libraries/pytorch_headers/torch/script.h) # Crucial, otherwise takes forever to compile


target_link_libraries(SKNativeTemplate ${SI_LINK_LIBS})
target_link_libraries(PoseModel PRIVATE
log StereoKitC 
libpytorch_jni.so
libfbjni.so
libtensorflowlite_jni.so 
libtensorflowlite_gpu_jni.so 
)

