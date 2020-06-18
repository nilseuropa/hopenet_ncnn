cmake_minimum_required(VERSION 2.8.3)
project(hopenet_ncnn)

add_compile_options(-std=c++11 -DUSE_AVX_INSTRUCTIONS=ON)

option(GPU_SUPPORT "Enable GPU support (Vulkan)" OFF)

find_package(ncnn REQUIRED)
if (${ncnn_FOUND})
  message("-- NCNN found.")
  message("-- NCNN_VULKAN flag is ${NCNN_VULKAN}")
  if (${NCNN_VULKAN})
    message("-- AUTO-ENABLING GPU_SUPPORT")
    set(GPU_SUPPORT ON)
  endif()
  include_directories(${ncnn_INCLUDE})
endif()

find_package(OpenCV REQUIRED COMPONENTS
  core highgui imgproc imgcodecs
)
include_directories(
  "include"
  ${OpenCV_INCLUDE_DIRS}
)

if (GPU_SUPPORT)
  message("-- GPU support is ENABLED")
  find_package(Vulkan) # REQUIRES ncnn to be built with vulkan
  if (${VULKAN_FOUND})
    message("-- Vulkan found.")
  else()
    message("-- ERROR: AUTO-DISABLING GPU_SUPPORT, because Vulkan was not found")
    set(GPU_SUPPORT OFF)
  endif()
else()
  message("-- GPU support is DISABLED")
endif()

configure_file("include/ncnn_config.h.in" "ncnn_config.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(hopenet_test src/hopenet.cpp src/ncnn_hopenet.cpp)
target_link_libraries(hopenet_test ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})
