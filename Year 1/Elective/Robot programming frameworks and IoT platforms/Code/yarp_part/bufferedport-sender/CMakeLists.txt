# set up our program
add_executable(bufferedport-sender)

# declare our source files
target_sources(bufferedport-sender PRIVATE main.cpp)

# link with YARP libraries
target_link_libraries(bufferedport-sender PRIVATE YARP::YARP_os
                                          	      YARP::YARP_init
                                                  YARP::YARP_sig)
