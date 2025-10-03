#!/bin/bash

if [[ "$1" == "clean" ]]; then rm -r ./build; exit 0; fi

#export CC=hipcc
#export CXX=hipcc
export CC=gcc
export CXX=g++

if [[ "$CC" == "hipcc" ]]; then
    ## Determine native GPU arch.
    touch empty.cpp
    arch=$(hipcc -E -dM empty.cpp | grep GFX | cut -d ' ' -f 2)
    rm -f empty.cpp
    export GPU_ARCH=$arch
    echo "Detected AMD GPU: $arch"
fi

mkdir -p build
if [[ -z "$1" ]];
then
    cd build && cmake .. && cmake --build .
else
    cd build && cmake .. && cmake --build . --target $1
fi
