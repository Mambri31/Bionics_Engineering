# set up our program
add_executable(simulator)

# declare our source files
target_sources(simulator PRIVATE main.cpp)

# link with YARP libraries
target_link_libraries(simulator PRIVATE YARP::YARP_os
                                          YARP::YARP_init
                                          YARP::YARP_sig
                                          YARP::YARP_dev)
