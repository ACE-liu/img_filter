# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/liuliu/projects/blur_check

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuliu/projects/blur_check/build

# Include any dependencies generated for this target.
include CMakeFiles/blur.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/blur.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/blur.dir/flags.make

CMakeFiles/blur.dir/filter_blur.cpp.o: CMakeFiles/blur.dir/flags.make
CMakeFiles/blur.dir/filter_blur.cpp.o: ../filter_blur.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liuliu/projects/blur_check/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/blur.dir/filter_blur.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/blur.dir/filter_blur.cpp.o -c /home/liuliu/projects/blur_check/filter_blur.cpp

CMakeFiles/blur.dir/filter_blur.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/blur.dir/filter_blur.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liuliu/projects/blur_check/filter_blur.cpp > CMakeFiles/blur.dir/filter_blur.cpp.i

CMakeFiles/blur.dir/filter_blur.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/blur.dir/filter_blur.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liuliu/projects/blur_check/filter_blur.cpp -o CMakeFiles/blur.dir/filter_blur.cpp.s

CMakeFiles/blur.dir/filter_blur.cpp.o.requires:

.PHONY : CMakeFiles/blur.dir/filter_blur.cpp.o.requires

CMakeFiles/blur.dir/filter_blur.cpp.o.provides: CMakeFiles/blur.dir/filter_blur.cpp.o.requires
	$(MAKE) -f CMakeFiles/blur.dir/build.make CMakeFiles/blur.dir/filter_blur.cpp.o.provides.build
.PHONY : CMakeFiles/blur.dir/filter_blur.cpp.o.provides

CMakeFiles/blur.dir/filter_blur.cpp.o.provides.build: CMakeFiles/blur.dir/filter_blur.cpp.o


CMakeFiles/blur.dir/haar_filter.cpp.o: CMakeFiles/blur.dir/flags.make
CMakeFiles/blur.dir/haar_filter.cpp.o: ../haar_filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liuliu/projects/blur_check/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/blur.dir/haar_filter.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/blur.dir/haar_filter.cpp.o -c /home/liuliu/projects/blur_check/haar_filter.cpp

CMakeFiles/blur.dir/haar_filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/blur.dir/haar_filter.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liuliu/projects/blur_check/haar_filter.cpp > CMakeFiles/blur.dir/haar_filter.cpp.i

CMakeFiles/blur.dir/haar_filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/blur.dir/haar_filter.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liuliu/projects/blur_check/haar_filter.cpp -o CMakeFiles/blur.dir/haar_filter.cpp.s

CMakeFiles/blur.dir/haar_filter.cpp.o.requires:

.PHONY : CMakeFiles/blur.dir/haar_filter.cpp.o.requires

CMakeFiles/blur.dir/haar_filter.cpp.o.provides: CMakeFiles/blur.dir/haar_filter.cpp.o.requires
	$(MAKE) -f CMakeFiles/blur.dir/build.make CMakeFiles/blur.dir/haar_filter.cpp.o.provides.build
.PHONY : CMakeFiles/blur.dir/haar_filter.cpp.o.provides

CMakeFiles/blur.dir/haar_filter.cpp.o.provides.build: CMakeFiles/blur.dir/haar_filter.cpp.o


# Object files for target blur
blur_OBJECTS = \
"CMakeFiles/blur.dir/filter_blur.cpp.o" \
"CMakeFiles/blur.dir/haar_filter.cpp.o"

# External object files for target blur
blur_EXTERNAL_OBJECTS =

libblur.a: CMakeFiles/blur.dir/filter_blur.cpp.o
libblur.a: CMakeFiles/blur.dir/haar_filter.cpp.o
libblur.a: CMakeFiles/blur.dir/build.make
libblur.a: CMakeFiles/blur.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liuliu/projects/blur_check/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libblur.a"
	$(CMAKE_COMMAND) -P CMakeFiles/blur.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/blur.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/blur.dir/build: libblur.a

.PHONY : CMakeFiles/blur.dir/build

CMakeFiles/blur.dir/requires: CMakeFiles/blur.dir/filter_blur.cpp.o.requires
CMakeFiles/blur.dir/requires: CMakeFiles/blur.dir/haar_filter.cpp.o.requires

.PHONY : CMakeFiles/blur.dir/requires

CMakeFiles/blur.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/blur.dir/cmake_clean.cmake
.PHONY : CMakeFiles/blur.dir/clean

CMakeFiles/blur.dir/depend:
	cd /home/liuliu/projects/blur_check/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuliu/projects/blur_check /home/liuliu/projects/blur_check /home/liuliu/projects/blur_check/build /home/liuliu/projects/blur_check/build /home/liuliu/projects/blur_check/build/CMakeFiles/blur.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/blur.dir/depend

