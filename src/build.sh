#!/bin/sh

set -e

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
CURRENT_FOLDER=`basename $SHELL_FOLDER`
cd $SHELL_FOLDER

# clear
if [ "$1" = "clear" ]; then
	rm -rf build
	rm Release/main -f
	exit 0
fi

# build
echo "Current working directory: $(pwd)"
rm -rf build
mkdir build
cd build
cmake -D ENABLE_FAST_MATH=ON -D WITH_TBB=ON -D WITH_OPENCL=ON -D ENABLE_NEON=ON -D ENABLE_AVX=ON .. 
make -j$(nproc)

# release
# The executable name 'main' is based on 'set(project_name main)' in CMakeLists.txt
chmod 777 main
mkdir -p "../Release"
mv main ../Release

# copy to Board if SYSROOT is set
if [ -n "$SYSROOT" ]; then
    echo "Copying to board..."
    mkdir -p $SYSROOT/userdata/Demo/$CURRENT_FOLDER
    if [ "$1" = "cpres" ]; then
        cp ../Release/* $SYSROOT/userdata/Demo/$CURRENT_FOLDER
    else
        cp ../Release/main $SYSROOT/userdata/Demo/$CURRENT_FOLDER
    fi
else
    echo "SYSROOT not set, skipping copy to board."
fi

echo "Build finished. Executable is in ../Release/main"
