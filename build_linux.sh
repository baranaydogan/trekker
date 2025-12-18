#!/bin/bash

# Linux build example

buildType=Release
buildShared=OFF
buildDir=build-static

# Modify these if you want autoencoder features
enableAE=OFF
Torch_DIR="/opt/libtorch/share/cmake/Torch"

mkdir -p ${buildDir}
cd ${buildDir}

cmakeExe=cmake
c_compiler=/bin/gcc
cxx_compiler=/bin/g++

${cmakeExe} \
-DCMAKE_C_COMPILER=${c_compiler} \
-DCMAKE_CXX_COMPILER=${cxx_compiler} \
-DCMAKE_BUILD_TYPE=${buildType} \
-DBUILD_SHARED_LIBS=${buildShared} \
-DENABLE_AUTOENCODER=${enableAE} \
-DTorch_DIR=${Torch_DIR} \
..


${cmakeExe} --build . --config ${buildType} --target install --parallel $(nproc)

cd ..
