# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /home/baran/apps/cmake-3.15.0-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/baran/apps/cmake-3.15.0-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/baran/Work/code/dev/baranaydogan/trekker

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/baran/Work/code/dev/baranaydogan/trekker/build/Linux

# Include any dependencies generated for this target.
include CMakeFiles/Trekker.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Trekker.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Trekker.dir/flags.make

# Object files for target Trekker
Trekker_OBJECTS =

# External object files for target Trekker
Trekker_EXTERNAL_OBJECTS = \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/base/algorithm.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/base/inputParser.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/base/streamline.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/base/tractogram.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/config/config_general.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/config/config_output.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/config/config_pathway.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/config/config_seeding.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/config/config_tracker.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/image/fod_image.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/image/image.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/image/roi_image.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/image/voxel.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/math/coordinate.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/math/doRandomThings.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/math/math_aux.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/math/sphericalHarmonics.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/nifticlib-2.0.0/niftilib/nifti1_io.c.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/nifticlib-2.0.0/znzlib/znzlib.c.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_defaults.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_estimatePosteriorMax.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_initialize.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_propagate.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_rejectionSample.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/streamline_local_probabilistic.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/tractogram_local_probabilistic.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/PTF.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_dataSupport.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_defaults.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_estimatePosteriorMax.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_initialize.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_propagate.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_rejectionSample.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/streamline_ptt.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/tractogram_ptt.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/tracker/tracker_thread.cpp.o" \
"/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/OBJS.dir/src/trekker.cpp.o"

libTrekker.so: CMakeFiles/OBJS.dir/src/base/algorithm.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/base/inputParser.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/base/streamline.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/base/tractogram.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/config/config_general.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/config/config_output.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/config/config_pathway.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/config/config_seeding.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/config/config_tracker.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/image/fod_image.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/image/image.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/image/roi_image.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/image/voxel.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/math/coordinate.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/math/doRandomThings.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/math/math_aux.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/math/sphericalHarmonics.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/nifticlib-2.0.0/niftilib/nifti1_io.c.o
libTrekker.so: CMakeFiles/OBJS.dir/src/nifticlib-2.0.0/znzlib/znzlib.c.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_defaults.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_estimatePosteriorMax.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_initialize.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_propagate.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/algorithm_local_probabilistic_rejectionSample.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/streamline_local_probabilistic.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/local_probabilistic/tractogram_local_probabilistic.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/PTF.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_dataSupport.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_defaults.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_estimatePosteriorMax.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_initialize.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_propagate.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/algorithm_ptt_rejectionSample.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/streamline_ptt.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/algorithms/ptt/tractogram_ptt.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/tracker/tracker_thread.cpp.o
libTrekker.so: CMakeFiles/OBJS.dir/src/trekker.cpp.o
libTrekker.so: CMakeFiles/Trekker.dir/build.make
libTrekker.so: CMakeFiles/Trekker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX shared library libTrekker.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Trekker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Trekker.dir/build: libTrekker.so

.PHONY : CMakeFiles/Trekker.dir/build

CMakeFiles/Trekker.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Trekker.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Trekker.dir/clean

CMakeFiles/Trekker.dir/depend:
	cd /home/baran/Work/code/dev/baranaydogan/trekker/build/Linux && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/baran/Work/code/dev/baranaydogan/trekker /home/baran/Work/code/dev/baranaydogan/trekker /home/baran/Work/code/dev/baranaydogan/trekker/build/Linux /home/baran/Work/code/dev/baranaydogan/trekker/build/Linux /home/baran/Work/code/dev/baranaydogan/trekker/build/Linux/CMakeFiles/Trekker.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Trekker.dir/depend
