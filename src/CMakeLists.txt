add_library(CVFunctions cvFunctions.cpp cvImageProcessing.cpp cvObjectDetection.cpp cvObjectTracking.cpp)

find_package(OpenCV REQUIRED)

include_directories(../include ../resources ${OpenCV_INCLUDE_DIRS})

add_executable(cvout visionMain.cpp)

target_link_libraries(cvout CVFunctions ${OpenCV_LIBS})