# set up our program
find_package(YARP REQUIRED)
add_executable(yarp-client)

# declare our source files
target_sources(yarp-client PRIVATE main.cpp)

# link with YARP libraries
target_link_libraries(yarp-client PRIVATE YARP::YARP_os
                        YARP::YARP_init
)