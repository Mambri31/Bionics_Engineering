# set up our program
add_executable(bufferedport-receiver)

# declare our source files
target_sources(bufferedport-receiver PRIVATE main.cpp)

# link with YARP libraries
target_link_libraries(bufferedport-receiver PRIVATE YARP::YARP_os
                                          	      YARP::YARP_init
                                                  YARP::YARP_sig)
