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
CMAKE_SOURCE_DIR = /home/guests/190111014/user-land-filesystem/fs/newfs

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/guests/190111014/user-land-filesystem/fs/newfs

# Include any dependencies generated for this target.
include CMakeFiles/newfs.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/newfs.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/newfs.dir/flags.make

CMakeFiles/newfs.dir/src/bitmap.c.o: CMakeFiles/newfs.dir/flags.make
CMakeFiles/newfs.dir/src/bitmap.c.o: src/bitmap.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guests/190111014/user-land-filesystem/fs/newfs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/newfs.dir/src/bitmap.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/newfs.dir/src/bitmap.c.o   -c /home/guests/190111014/user-land-filesystem/fs/newfs/src/bitmap.c

CMakeFiles/newfs.dir/src/bitmap.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/newfs.dir/src/bitmap.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/guests/190111014/user-land-filesystem/fs/newfs/src/bitmap.c > CMakeFiles/newfs.dir/src/bitmap.c.i

CMakeFiles/newfs.dir/src/bitmap.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/newfs.dir/src/bitmap.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/guests/190111014/user-land-filesystem/fs/newfs/src/bitmap.c -o CMakeFiles/newfs.dir/src/bitmap.c.s

CMakeFiles/newfs.dir/src/newfs.c.o: CMakeFiles/newfs.dir/flags.make
CMakeFiles/newfs.dir/src/newfs.c.o: src/newfs.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guests/190111014/user-land-filesystem/fs/newfs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/newfs.dir/src/newfs.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/newfs.dir/src/newfs.c.o   -c /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs.c

CMakeFiles/newfs.dir/src/newfs.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/newfs.dir/src/newfs.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs.c > CMakeFiles/newfs.dir/src/newfs.c.i

CMakeFiles/newfs.dir/src/newfs.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/newfs.dir/src/newfs.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs.c -o CMakeFiles/newfs.dir/src/newfs.c.s

CMakeFiles/newfs.dir/src/newfs_debug.c.o: CMakeFiles/newfs.dir/flags.make
CMakeFiles/newfs.dir/src/newfs_debug.c.o: src/newfs_debug.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guests/190111014/user-land-filesystem/fs/newfs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/newfs.dir/src/newfs_debug.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/newfs.dir/src/newfs_debug.c.o   -c /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs_debug.c

CMakeFiles/newfs.dir/src/newfs_debug.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/newfs.dir/src/newfs_debug.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs_debug.c > CMakeFiles/newfs.dir/src/newfs_debug.c.i

CMakeFiles/newfs.dir/src/newfs_debug.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/newfs.dir/src/newfs_debug.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs_debug.c -o CMakeFiles/newfs.dir/src/newfs_debug.c.s

CMakeFiles/newfs.dir/src/newfs_utils.c.o: CMakeFiles/newfs.dir/flags.make
CMakeFiles/newfs.dir/src/newfs_utils.c.o: src/newfs_utils.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guests/190111014/user-land-filesystem/fs/newfs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/newfs.dir/src/newfs_utils.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/newfs.dir/src/newfs_utils.c.o   -c /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs_utils.c

CMakeFiles/newfs.dir/src/newfs_utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/newfs.dir/src/newfs_utils.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs_utils.c > CMakeFiles/newfs.dir/src/newfs_utils.c.i

CMakeFiles/newfs.dir/src/newfs_utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/newfs.dir/src/newfs_utils.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/guests/190111014/user-land-filesystem/fs/newfs/src/newfs_utils.c -o CMakeFiles/newfs.dir/src/newfs_utils.c.s

# Object files for target newfs
newfs_OBJECTS = \
"CMakeFiles/newfs.dir/src/bitmap.c.o" \
"CMakeFiles/newfs.dir/src/newfs.c.o" \
"CMakeFiles/newfs.dir/src/newfs_debug.c.o" \
"CMakeFiles/newfs.dir/src/newfs_utils.c.o"

# External object files for target newfs
newfs_EXTERNAL_OBJECTS =

newfs: CMakeFiles/newfs.dir/src/bitmap.c.o
newfs: CMakeFiles/newfs.dir/src/newfs.c.o
newfs: CMakeFiles/newfs.dir/src/newfs_debug.c.o
newfs: CMakeFiles/newfs.dir/src/newfs_utils.c.o
newfs: CMakeFiles/newfs.dir/build.make
newfs: /usr/lib/x86_64-linux-gnu/libfuse.so
newfs: /home/guests/190111014/lib/libddriver.a
newfs: CMakeFiles/newfs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/guests/190111014/user-land-filesystem/fs/newfs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking C executable newfs"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/newfs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/newfs.dir/build: newfs

.PHONY : CMakeFiles/newfs.dir/build

CMakeFiles/newfs.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/newfs.dir/cmake_clean.cmake
.PHONY : CMakeFiles/newfs.dir/clean

CMakeFiles/newfs.dir/depend:
	cd /home/guests/190111014/user-land-filesystem/fs/newfs && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/guests/190111014/user-land-filesystem/fs/newfs /home/guests/190111014/user-land-filesystem/fs/newfs /home/guests/190111014/user-land-filesystem/fs/newfs /home/guests/190111014/user-land-filesystem/fs/newfs /home/guests/190111014/user-land-filesystem/fs/newfs/CMakeFiles/newfs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/newfs.dir/depend

