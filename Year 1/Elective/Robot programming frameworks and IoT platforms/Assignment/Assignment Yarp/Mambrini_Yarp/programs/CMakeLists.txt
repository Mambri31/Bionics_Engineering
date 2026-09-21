cmake_minimum_required(VERSION 3.12)

project(Mambrini_Yarp)

find_package(YARP REQUIRED)



add_executable(program1 program1.cpp)

target_link_libraries(program1 PRIVATE 
                        YARP::YARP_os
                        YARP::YARP_init 
                        YARP::YARP_sig
                        YARP::YARP_dev
                        )


add_executable(program2 program2.cpp)


target_link_libraries(program2 PRIVATE 
                        YARP::YARP_os
                        YARP::YARP_init 
                        YARP::YARP_sig
                        YARP::YARP_dev
                        )
