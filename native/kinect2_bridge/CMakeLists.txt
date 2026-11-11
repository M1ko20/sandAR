cmake_minimum_required(VERSION 3.16)
project(kinect2_bridge CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# libfreenect2 se instaluje do third_party/install (viz scripts/install_kinect2.sh)
find_package(freenect2 REQUIRED CONFIG)

add_executable(kinect2_bridge main.cpp)
target_include_directories(kinect2_bridge PRIVATE ${freenect2_INCLUDE_DIRS})
target_link_libraries(kinect2_bridge PRIVATE ${freenect2_LIBRARIES})

# Aby binárka našla libfreenect2 i bez nastavování DYLD_/LD_LIBRARY_PATH.
get_filename_component(_fn2_libdir "${freenect2_LIBRARY}" DIRECTORY)
set_target_properties(kinect2_bridge PROPERTIES
  BUILD_RPATH "${_fn2_libdir}"
  INSTALL_RPATH "${_fn2_libdir}")
