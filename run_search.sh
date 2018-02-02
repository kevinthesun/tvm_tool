#!/bin/bash
echo "Searching mobilenet..."
for i in {12..21} ;
do
    ( TVM_NUM_THREADS=4 python conv2d_nchw_x86_search.py --index $i --target "llvm -mcpu=skylake-avx512" & )
done
