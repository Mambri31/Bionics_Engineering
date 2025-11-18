# set up our program
add_executable(port-receiver)

# declare our source files
target_sources(port-receiver PRIVATE main.cpp)

# link with YARP libraries
target_link_libraries(port-receiver PRIVATE
    YARP::YARP_os
    YARP::YARP_init
)