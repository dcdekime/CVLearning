# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ddekime/CVLearning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ddekime/CVLearning/build

# Include any dependencies generated for this target.
include src/CMakeFiles/cvout.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cvout.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cvout.dir/flags.make

src/CMakeFiles/cvout.dir/visionMain.cpp.o: src/CMakeFiles/cvout.dir/flags.make
src/CMakeFiles/cvout.dir/visionMain.cpp.o: ../src/visionMain.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ddekime/CVLearning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/cvout.dir/visionMain.cpp.o"
	cd /home/ddekime/CVLearning/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvout.dir/visionMain.cpp.o -c /home/ddekime/CVLearning/src/visionMain.cpp

src/CMakeFiles/cvout.dir/visionMain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvout.dir/visionMain.cpp.i"
	cd /home/ddekime/CVLearning/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ddekime/CVLearning/src/visionMain.cpp > CMakeFiles/cvout.dir/visionMain.cpp.i

src/CMakeFiles/cvout.dir/visionMain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvout.dir/visionMain.cpp.s"
	cd /home/ddekime/CVLearning/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ddekime/CVLearning/src/visionMain.cpp -o CMakeFiles/cvout.dir/visionMain.cpp.s

# Object files for target cvout
cvout_OBJECTS = \
"CMakeFiles/cvout.dir/visionMain.cpp.o"

# External object files for target cvout
cvout_EXTERNAL_OBJECTS =

src/cvout: src/CMakeFiles/cvout.dir/visionMain.cpp.o
src/cvout: src/CMakeFiles/cvout.dir/build.make
src/cvout: src/libCVFunctions.a
src/cvout: /usr/local/lib/libopencv_dnn.so.3.4.14
src/cvout: /usr/local/lib/libopencv_highgui.so.3.4.14
src/cvout: /usr/local/lib/libopencv_ml.so.3.4.14
src/cvout: /usr/local/lib/libopencv_objdetect.so.3.4.14
src/cvout: /usr/local/lib/libopencv_shape.so.3.4.14
src/cvout: /usr/local/lib/libopencv_stitching.so.3.4.14
src/cvout: /usr/local/lib/libopencv_superres.so.3.4.14
src/cvout: /usr/local/lib/libopencv_videostab.so.3.4.14
src/cvout: /usr/local/lib/libopencv_viz.so.3.4.14
src/cvout: /usr/local/lib/libopencv_calib3d.so.3.4.14
src/cvout: /usr/local/lib/libopencv_features2d.so.3.4.14
src/cvout: /usr/local/lib/libopencv_flann.so.3.4.14
src/cvout: /usr/local/lib/libopencv_photo.so.3.4.14
src/cvout: /usr/local/lib/libopencv_video.so.3.4.14
src/cvout: /usr/local/lib/libopencv_videoio.so.3.4.14
src/cvout: /usr/local/lib/libopencv_imgcodecs.so.3.4.14
src/cvout: /usr/local/lib/libopencv_imgproc.so.3.4.14
src/cvout: /usr/local/lib/libopencv_core.so.3.4.14
src/cvout: src/CMakeFiles/cvout.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ddekime/CVLearning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cvout"
	cd /home/ddekime/CVLearning/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvout.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cvout.dir/build: src/cvout

.PHONY : src/CMakeFiles/cvout.dir/build

src/CMakeFiles/cvout.dir/clean:
	cd /home/ddekime/CVLearning/build/src && $(CMAKE_COMMAND) -P CMakeFiles/cvout.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cvout.dir/clean

src/CMakeFiles/cvout.dir/depend:
	cd /home/ddekime/CVLearning/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ddekime/CVLearning /home/ddekime/CVLearning/src /home/ddekime/CVLearning/build /home/ddekime/CVLearning/build/src /home/ddekime/CVLearning/build/src/CMakeFiles/cvout.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cvout.dir/depend

