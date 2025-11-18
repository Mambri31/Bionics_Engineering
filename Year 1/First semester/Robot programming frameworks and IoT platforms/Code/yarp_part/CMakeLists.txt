# YARP needs CMake 3.0 or greater
cmake_minimum_required(VERSION 3.12)

project(yarp_exercises)

# executables will all be put be in bin folder
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# find YARP
find_package(YARP COMPONENTS os sig dev REQUIRED)

###################################################
# add program folders
# should not be changed, unless it doesen't work :)

# find all subdirectories
macro(subdirlist result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()
subdirlist(SUBDIRS ${CMAKE_SOURCE_DIR})

# remove build directory
#list(REMOVE_ITEM SUBDIRS "build")
get_filename_component(BUILD_DIR "${CMAKE_BINARY_DIR}" NAME)
list(REMOVE_ITEM SUBDIRS "${BUILD_DIR}")

# add all good folders
foreach(subdir ${SUBDIRS})
  add_subdirectory(${subdir})
endforeach()
###################################################