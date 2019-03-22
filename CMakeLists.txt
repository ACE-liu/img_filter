cmake_minimum_required(VERSION 2.8)

set(prj_name "blur")
project(${prj_name})
add_compile_options(-std=c++11)

include_directories(include/)


set(lib "")


find_package(OpenCV 3.3.1 REQUIRED)
if(OpenCV_FOUND)
#     set(USE_OPENCV true)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    list(APPEND lib ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
    message(WARNING "OpenCV not found")
endif(OpenCV_FOUND)

find_package(Boost 1.61.0 REQUIRED COMPONENTS thread chrono filesystem system date_time)
if(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    list(APPEND lib ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(FATAL_ERROR "Need Boost to be included!")
endif(Boost_FOUND)


file(GLOB hdrs include/*.h)
file(GLOB srcs filter_blur.cpp haar_filter.cpp)

add_library(${prj_name} ${srcs} ${hdrs})
target_link_libraries(${prj_name}  ${lib})

add_executable(${prj_name}check main.cpp)
target_link_libraries(${prj_name}check ${prj_name} ${lib})
#add_executable(haarcheck haar_filter.cpp)
#target_link_libraries(haarcheck  ${lib})
