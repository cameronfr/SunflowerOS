cmake_minimum_required(VERSION 3.11)


# Add python includes
include_directories("/data/data/com.termux/files/usr/include/python3.10")
# Add python libraries
link_directories("/data/data/com.termux/files/usr/lib")


add_executable( kernelLaunchEx
  main.cpp
)
add_library( kernelLaunch SHARED
  main.cpp
)

# ldd python3 -> python3.10, but need to link executable with python3.10 otherwise symbols not found?
target_link_libraries(kernelLaunchEx python3.10 log)
target_link_libraries(kernelLaunch python3 log)