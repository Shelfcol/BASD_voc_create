cmake_minimum_required( VERSION 2.8 )
project( voc_create )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
list(APPEND CMAKE_PREFIX_PATH "/home/gxf/software/libtorch_ABI_1/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# opencv 
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )




add_executable( feature_training src/feature_training.cpp )
target_link_libraries( feature_training  ${OpenCV_LIBS}  ${DBoW3_LIBS}  ${TORCH_LIBRARIES} )

add_executable( loop_closure src/loop_closure.cpp )
target_link_libraries( loop_closure ${OpenCV_LIBS}  ${DBoW3_LIBS}  ${TORCH_LIBRARIES} )

add_executable( gen_vocab src/gen_vocab_large.cpp )
target_link_libraries( gen_vocab ${OpenCV_LIBS}  ${DBoW3_LIBS}  ${TORCH_LIBRARIES} )

add_executable( fast src/fast.cpp )
target_link_libraries( fast ${OpenCV_LIBS}  ${DBoW3_LIBS}  ${TORCH_LIBRARIES} )

add_executable( orb_match src/orb_match.cpp )
target_link_libraries( orb_match ${OpenCV_LIBS}  ${DBoW3_LIBS}  ${TORCH_LIBRARIES} )

add_executable( BASD_matching src/BASD_matching.cpp )
target_link_libraries( BASD_matching ${OpenCV_LIBS}  ${DBoW3_LIBS}  ${TORCH_LIBRARIES} )
