# set up our program
add_executable(port-sender)

# declare our source files
target_sources(port-sender PRIVATE main.cpp)

# link with YARP libraries
target_link_libraries(port-sender PRIVATE YARP::YARP_os
                                          YARP::YARP_init)
