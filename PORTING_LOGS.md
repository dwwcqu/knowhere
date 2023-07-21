# knowhere 移植日志

## 2023/6/16

1. **knowhere**本身使用的 **cuda** 接口较少，主要是**依赖其他库**较多；

2. 与 **cuda** 相关的依赖库有：<u>*rapidsai/raft*</u>、<u>*rapidsai/rmm*</u>、<u>*nvidia/thrust*</u>、*<u>nvidia/cutlass</u>*、*<u>libcurand.so</u>*、<u>*libcusolver.so*</u>、<u>*libcublas.so*</u>、<u>*libculibos.a*</u>、<u>*libcublasLt.so*</u>、<u>*libcusparse.so*</u>；这里主要是 **knowhere** 依赖 <u>*rapidsai/raft*</u>、<u>*rapidsai/rmm*</u>、<u>*nvidia/thrust*</u> 、*<u>nvidia/cutlass</u>* 四个库，从依赖了 **cuda** 的底层共享库：*<u>nvidia/cutlass</u>*、*<u>libcurand.so</u>*、<u>*libcusolver.so*</u>、<u>*libcublas.so*</u>、<u>*libculibos.a*</u>、<u>*libcublasLt.so*</u>、<u>*libcusparse.so*</u>；

3. 目前，对于*<u>libcurand.so</u>*、*<u>libcusolver.so</u>*、*<u>libcusparse.so</u>*、<u>*nvidia/thrust*</u> 、<u>*libcublas.so*</u> 的依赖，在ROCm平台下有相应的库支持：

   + *<u>libcurand.so</u>*  <------> *<u>rocRAND</u>*
   + *<u>libcusolver.so</u>*  <------> *<u>rocSOLVER</u>*
   + *<u>libcusparse.so</u>*  <------> <u>*rocSPARSE*</u>
   + <u>*nvidia/thrust*</u>  <------> *<u>ROCm/rocThrust</u>*
   + <u>*libcublas.so*</u>  <-----> <u>*rocBLAS*</u>

   目前，在 ROCm 平台下，与 <u>*libcublasLt.so*</u>、<u>*libculibos.a*</u> 相适配的库，还未找到。其中的 *<u>nvidia/cutlass</u>* 也是依赖底层库，而 <u>*libcublasLt.so*</u>、<u>*libculibos.a*</u> 两个库还不确定。

4. 如果在 ROCm 平台下，没有与 <u>*libcublasLt.so*</u>、<u>*libculibos.a*</u> 相适配的库的话，**移植过程可能会无法成功**；对于是否有与之相适配的库，后续需要时间去仔细调研一下。

5. 下面附上在 **N 卡**上编译生成 GPU 版本的 `libknowhere.so` 库的**链接信息**(格式化了以下，便于阅读)：

   ```bash
   /usr/bin/c++ 
   -fPIC -Wall -fPIC -std=gnu++17 -m64 -fopenmp -O0 -g -m64 -shared
   -Wl,-soname,libknowhere.so 
   -o libknowhere.so 
   CMakeFiles/knowhere.dir/src/common/comp/brute_force.cc.o 
   CMakeFiles/knowhere.dir/src/common/comp/knowhere_config.cc.o 
   CMakeFiles/knowhere.dir/src/common/comp/time_recorder.cc.o 
   CMakeFiles/knowhere.dir/src/common/config.cc.o
   CMakeFiles/knowhere.dir/src/common/factory.cc.o
   CMakeFiles/knowhere.dir/src/common/log.cc.o
   CMakeFiles/knowhere.dir/src/common/prometheus_client.cc.o
   CMakeFiles/knowhere.dir/src/common/range_util.cc.o
   CMakeFiles/knowhere.dir/src/common/utils.cc.o
   CMakeFiles/knowhere.dir/src/index/flat/flat.cc.o
   CMakeFiles/knowhere.dir/src/index/hnsw/hnsw.cc.o
   CMakeFiles/knowhere.dir/src/index/ivf/ivf.cc.o
   CMakeFiles/knowhere.dir/src/index/ivf_raft/ivf_raft.cu.o
   CMakeFiles/knowhere.dir/src/io/FaissIO.cc.o   
   -L/usr/local/cuda-11.6/lib64  
   -L/usr/local/cuda-11.6/targets/x86_64-linux/lib/stubs  
   -L/usr/local/cuda-11.6/targets/x86_64-linux/lib  
   -Wl,-rpath,/usr/local/cuda-11.6/lib64:/usr/local/cuda-11.6/targets/x86_64-linux/lib: 
   libfaiss.a 
   /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so 
   /usr/lib/x86_64-linux-gnu/libpthread.so 
   /usr/lib/x86_64-linux-gnu/libopenblas.so 
   -lm -ldl 
   libknowhere_utils.a 
   /home/qnx/.conan/data/glog/0.6.0/_/_/package/be570b961190cbdda5166d31ebc655fff1a0dd60/lib/libglogd.a 
   /home/qnx/.conan/data/libunwind/1.6.2/_/_/package/9fe0223acd399813a540a894f2773f5c2b75c565/lib/libunwind-coredump.a 
   /home/qnx/.conan/data/libunwind/1.6.2/_/_/package/9fe0223acd399813a540a894f2773f5c2b75c565/lib/libunwind-setjmp.a 
   /home/qnx/.conan/data/libunwind/1.6.2/_/_/package/9fe0223acd399813a540a894f2773f5c2b75c565/lib/libunwind-ptrace.a 
   /home/qnx/.conan/data/libunwind/1.6.2/_/_/package/9fe0223acd399813a540a894f2773f5c2b75c565/lib/libunwind-generic.a 
   /home/qnx/.conan/data/libunwind/1.6.2/_/_/package/9fe0223acd399813a540a894f2773f5c2b75c565/lib/libunwind.a 
   /home/qnx/.conan/data/xz_utils/5.4.0/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib/liblzma.a 
   /home/qnx/.conan/data/prometheus-cpp/1.1.0/_/_/package/3f935e47ad9d32b7c980128de227d21fbf1c4979/lib/libprometheus-cpp-push.a 
   /home/qnx/.conan/data/prometheus-cpp/1.1.0/_/_/package/3f935e47ad9d32b7c980128de227d21fbf1c4979/lib/libprometheus-cpp-core.a 
   /home/qnx/.conan/data/libcurl/8.0.1/_/_/package/f0fb7ac9cb80bf1caef3ccb84335797726139fb8/lib/libcurl.a 
   /home/qnx/.conan/data/openssl/1.1.1t/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib/libssl.a 
   /home/qnx/.conan/data/openssl/1.1.1t/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib/libcrypto.a 
   -lpthread -lrt 
   /home/qnx/.conan/data/zlib/1.2.13/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib/libz.a 
   /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcudart.so 
   -lpthread -ldl 
   /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcurand.so 
   /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusolver.so 
   /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcublas.so 
   /usr/local/cuda-11.6/targets/x86_64-linux/lib/libculibos.a
   /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcublasLt.so 
   /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusparse.so 
   -lcudadevrt -lcudart_static -lrt -lpthread -ldl 
   ```

6. 总结：从上面分析来讲，为了成功编译 GPU 版本的 **knowhere** 库，需要移植的依赖库有以下几个：<u>*rapidsai/raft*</u>、<u>*rapidsai/rmm*</u>、*<u>nvidia/cutlass</u>* ，如果 *<u>nvidia/cutlass</u>* 在 ROCm 平台下有相对应支持的库，也就只需要移植 <u>*rapidsai/raft*</u>、<u>*rapidsai/rmm*</u>，这两个库没有在 AMD GPU 下的版本。后续移植任务，需要先将 <u>*rapidsai/raft*</u>、<u>*rapidsai/rmm*</u>、*<u>nvidia/cutlass</u>* 移植成功先。

## 2023/6/19

+ 进一步调研发现，在编译 *knowhere* 的 *GPU* 版本时，还需要依赖 `libfaiss` 库，故而需要移植项目中的 `thirdparty/faiss/gpu` 目录下的接口；

+ 在 *NVIDIA* 环境下编译 `libfaiss` 库， 其链接过程如下：

  ```bash
  /usr/bin/ar qc libfaiss.a
  CMakeFiles/faiss.dir/AutoTune.cpp.o 
  CMakeFiles/faiss.dir/Clustering.cpp.o 
  CMakeFiles/faiss.dir/IVFlib.cpp.o 
  CMakeFiles/faiss.dir/Index.cpp.o 
  CMakeFiles/faiss.dir/Index2Layer.cpp.o 
  CMakeFiles/faiss.dir/IndexAdditiveQuantizer.cpp.o 
  CMakeFiles/faiss.dir/IndexBinary.cpp.o 
  CMakeFiles/faiss.dir/IndexBinaryFlat.cpp.o 
  CMakeFiles/faiss.dir/IndexBinaryFromFloat.cpp.o 
  CMakeFiles/faiss.dir/IndexBinaryHNSW.cpp.o 
  CMakeFiles/faiss.dir/IndexBinaryHash.cpp.o 
  CMakeFiles/faiss.dir/IndexBinaryIVF.cpp.o 
  CMakeFiles/faiss.dir/IndexFlat.cpp.o 
  CMakeFiles/faiss.dir/IndexFlatCodes.cpp.o 
  CMakeFiles/faiss.dir/IndexHNSW.cpp.o 
  CMakeFiles/faiss.dir/IndexIDMap.cpp.o 
  CMakeFiles/faiss.dir/IndexIVF.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFAdditiveQuantizer.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFFlat.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFPQ.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFFastScan.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFAdditiveQuantizerFastScan.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFPQFastScan.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFPQR.cpp.o 
  CMakeFiles/faiss.dir/IndexIVFSpectralHash.cpp.o 
  CMakeFiles/faiss.dir/IndexLSH.cpp.o 
  CMakeFiles/faiss.dir/IndexNNDescent.cpp.o 
  CMakeFiles/faiss.dir/IndexLattice.cpp.o 
  CMakeFiles/faiss.dir/IndexNSG.cpp.o 
  CMakeFiles/faiss.dir/IndexPQ.cpp.o 
  CMakeFiles/faiss.dir/IndexFastScan.cpp.o 
  CMakeFiles/faiss.dir/IndexAdditiveQuantizerFastScan.cpp.o 
  CMakeFiles/faiss.dir/IndexPQFastScan.cpp.o 
  CMakeFiles/faiss.dir/IndexPreTransform.cpp.o 
  CMakeFiles/faiss.dir/IndexRefine.cpp.o 
  CMakeFiles/faiss.dir/IndexReplicas.cpp.o 
  CMakeFiles/faiss.dir/IndexRowwiseMinMax.cpp.o 
  CMakeFiles/faiss.dir/IndexScalarQuantizer.cpp.o 
  CMakeFiles/faiss.dir/IndexShards.cpp.o 
  CMakeFiles/faiss.dir/MatrixStats.cpp.o 
  CMakeFiles/faiss.dir/MetaIndexes.cpp.o 
  CMakeFiles/faiss.dir/VectorTransform.cpp.o 
  CMakeFiles/faiss.dir/clone_index.cpp.o 
  CMakeFiles/faiss.dir/index_factory.cpp.o 
  CMakeFiles/faiss.dir/impl/AuxIndexStructures.cpp.o 
  CMakeFiles/faiss.dir/impl/IDSelector.cpp.o 
  CMakeFiles/faiss.dir/impl/FaissException.cpp.o 
  CMakeFiles/faiss.dir/impl/HNSW.cpp.o 
  CMakeFiles/faiss.dir/impl/NSG.cpp.o 
  CMakeFiles/faiss.dir/impl/PolysemousTraining.cpp.o 
  CMakeFiles/faiss.dir/impl/ProductQuantizer.cpp.o 
  CMakeFiles/faiss.dir/impl/AdditiveQuantizer.cpp.o 
  CMakeFiles/faiss.dir/impl/ResidualQuantizer.cpp.o 
  CMakeFiles/faiss.dir/impl/LocalSearchQuantizer.cpp.o 
  CMakeFiles/faiss.dir/impl/ProductAdditiveQuantizer.cpp.o 
  CMakeFiles/faiss.dir/impl/ScalarQuantizer.cpp.o 
  CMakeFiles/faiss.dir/impl/index_read.cpp.o 
  CMakeFiles/faiss.dir/impl/index_write.cpp.o 
  CMakeFiles/faiss.dir/impl/io.cpp.o 
  CMakeFiles/faiss.dir/impl/kmeans1d.cpp.o 
  CMakeFiles/faiss.dir/impl/lattice_Zn.cpp.o 
  CMakeFiles/faiss.dir/impl/pq4_fast_scan.cpp.o 
  CMakeFiles/faiss.dir/impl/pq4_fast_scan_search_1.cpp.o 
  CMakeFiles/faiss.dir/impl/pq4_fast_scan_search_qbs.cpp.o 
  CMakeFiles/faiss.dir/impl/NNDescent.cpp.o 
  CMakeFiles/faiss.dir/invlists/BlockInvertedLists.cpp.o 
  CMakeFiles/faiss.dir/invlists/DirectMap.cpp.o 
  CMakeFiles/faiss.dir/invlists/InvertedLists.cpp.o 
  CMakeFiles/faiss.dir/invlists/InvertedListsIOHook.cpp.o 
  CMakeFiles/faiss.dir/utils/Heap.cpp.o 
  CMakeFiles/faiss.dir/utils/WorkerThread.cpp.o 
  CMakeFiles/faiss.dir/utils/distances.cpp.o 
  CMakeFiles/faiss.dir/utils/distances_simd.cpp.o 
  CMakeFiles/faiss.dir/utils/extra_distances.cpp.o 
  CMakeFiles/faiss.dir/utils/hamming.cpp.o 
  CMakeFiles/faiss.dir/utils/partitioning.cpp.o 
  CMakeFiles/faiss.dir/utils/quantize_lut.cpp.o 
  CMakeFiles/faiss.dir/utils/random.cpp.o 
  CMakeFiles/faiss.dir/utils/utils.cpp.o 
  CMakeFiles/faiss.dir/invlists/OnDiskInvertedLists.cpp.o 
  CMakeFiles/faiss.dir/gpu/GpuAutoTune.cpp.o 
  CMakeFiles/faiss.dir/gpu/GpuCloner.cpp.o 
  CMakeFiles/faiss.dir/gpu/GpuClonerOptions.cpp.o 
  CMakeFiles/faiss.dir/gpu/GpuDistance.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIcmEncoder.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIndex.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIndexBinaryFlat.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIndexFlat.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIndexIVF.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIndexIVFFlat.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIndexIVFPQ.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuIndexIVFScalarQuantizer.cu.o 
  CMakeFiles/faiss.dir/gpu/GpuResources.cpp.o 
  CMakeFiles/faiss.dir/gpu/StandardGpuResources.cpp.o 
  CMakeFiles/faiss.dir/gpu/impl/BinaryDistance.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/BinaryFlatIndex.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/BroadcastSum.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/Distance.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/FlatIndex.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IndexUtils.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFAppend.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFBase.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFFlat.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFFlatScan.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFInterleaved.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFPQ.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFUtils.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFUtilsSelect1.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IVFUtilsSelect2.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/InterleavedCodes.cpp.o 
  CMakeFiles/faiss.dir/gpu/impl/L2Norm.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/L2Select.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/PQScanMultiPassPrecomputed.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/RemapIndices.cpp.o 
  CMakeFiles/faiss.dir/gpu/impl/VectorResidual.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved1.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved32.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved64.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved128.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved256.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved512.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved1024.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/scan/IVFInterleaved2048.cu.o 
  CMakeFiles/faiss.dir/gpu/impl/IcmEncoder.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/BlockSelectFloat.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/DeviceUtils.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/StackDeviceMemory.cpp.o 
  CMakeFiles/faiss.dir/gpu/utils/Timer.cpp.o 
  CMakeFiles/faiss.dir/gpu/utils/WarpSelectFloat.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat1.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat32.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat64.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat128.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloat256.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatF512.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatF1024.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatF2048.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatT512.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatT1024.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/blockselect/BlockSelectFloatT2048.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat1.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat32.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat64.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat128.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloat256.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatF512.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatF1024.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatF2048.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatT512.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatT1024.cu.o 
  CMakeFiles/faiss.dir/gpu/utils/warpselect/WarpSelectFloatT2048.cu.o
  /usr/bin/ranlib libfaiss.a
  ```

## 2023/6/20

+ `faiss` 的 *GPU* 版本向 *ROCm* 的接口移植工作已经完成，今天的任务就是修改构建配置，修改文件名后缀，完成整个 `faiss` 的构建过程；
+ *CUDA* 中的 `half` 类型和 *HIP* 中的 `__half` 类型在访问成员变量 `__x` 的方式上存在不同，在 *CUDA* 中 `x` 是公共成员，在 *HIP* 中是保护成员；
+ 关于转码 `faiss/gpu/utils/PtxUtils.h` 中的 `asm()` 命令时存在问题，在 *HIP* 中不存在 *PTX* 中间码的概念，这里能够行得通的转码方式，就是将该 *PTX* 代码理解清楚后，将其转为 *ROCm* GPU 下的相应设备指令。因此，这里存在一个额外的任务，就是学习 *CUDA* 中的 *PTX* 并行编程；

## 2023/6/21

```c++
	#ifdef __HIP_PLATFORM_NVIDIA__
    static_assert(kWarpSize == 32, "unexpected warp size");
    #else
    static_assert(kWarpSize == 64, "unexpected warp size");
    #endif
```

+ `faiss/gpu/utils/PtxUtils.h` 文件中的 *PTX* 代码向 *GCN ISA* 指令转码。在转码的学习过程中发现，部分 *CUDA PTX* 指令能在 *GCN ISA* 下找到相应的指令(但是这是不确定的，因为这两者之间不存在完全一对一的功能)，故而要么用 `__device__` 代码实现该 *PTX* 指令，要么就是使用 *GCN ISA* 指令实现同样的功能；

+ 在进一步转码过程中发现，`gpu/impl/PQCodeLoad.h` 中也存在大量的 *PTX* 代码，同样的还有：`gpu/utils/LoadStoreOperators.h`

+ `faiss` 在 *kernel* 函数或 *device* 函数的实现上，使用了大量的 `template` 语法，在 `hipcc` 编译 `template` 的 `__global__`、`__device__` 函数时，存在的模板替换失败的情况，需要手动解决；

+ 在转码编译过程中出现的新问题：

  ```bash
  /home/dengww/Ports/faiss/faiss/gpu/impl/PQScanMultiPassPrecomputed.cpp:31:17: error: illegal SGPR to VGPR copy
  __global__ void pqScanPrecomputedInterleaved(
                  ^
  fatal error: error in backend: Not supported instr: <MCInst 1815 <MCOperand Reg:344> <MCOperand Reg:479>>
  clang-13: error: clang frontend command failed with exit code 70 (use -v to see invocation)
  clang version 13.0.0 (/buildspace/das-build/zifang-4.5.2/llvm-project/clang 4fbbb57356b98000fde3816fd8b20aab5837974b)
  Target: x86_64-unknown-linux-gnu
  Thread model: posix
  InstalledDir: /opt/rocm-4.5.2/llvm/bin
  clang-13: note: diagnostic msg: Error generating preprocessed source(s).
  ```

  该问题目前还无法解决，暂时保留！在后续编译时，同样出现了以上相似问题：

  ```bash	
  In file included from /home/dengww/Ports/faiss/faiss/gpu/impl/scan/IVFInterleaved1.cpp:8:
  In file included from /home/dengww/Ports/faiss/faiss/gpu/impl/scan/IVFInterleavedImpl.h:10:
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  __global__ void ivfInterleavedScan(
                  ^
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  /home/dengww/Ports/faiss/faiss/gpu/impl/IVFInterleaved.h:41:17: error: illegal SGPR to VGPR copy
  fatal error: too many errors emitted, stopping now [-ferror-limit=]
  clang-13: error: clang frontend command failed with exit code 70 (use -v to see invocation)
  clang version 13.0.0 (/buildspace/das-build/zifang-4.5.2/llvm-project/clang 4fbbb57356b98000fde3816fd8b20aab5837974b)
  Target: x86_64-unknown-linux-gnu
  Thread model: posix
  InstalledDir: /opt/rocm-4.5.2/llvm/bin
  clang-13: note: diagnostic msg: Error generating preprocessed source(s).
  ```

## 2023/6/25

+ 着手解决上述 `error: illegal SGPR to VGPR copy` 的错误；
+ 记录一个编译进度：目前 `faiss` 库除了上面描述的两个问题：1、*PTX* 代码的等效代码转换；2、`error: illegal SGPR to VGPR copy`；其他所有 `faiss` 库的源码部分都能够正确编译。
+ 前一个问题需要对每个 *PTX* 代码进行 *HIP* 实现；第二个问题，我目前没有任何解决办法，再向其他人员寻求帮助；

## 2023/6/26

+ 学习 GCN ISA 指令集架构的内容，便于转 *PTX* 代码；

## 2023/6/27

+ 学习 *PTX ISA* 和 *GCN ISA* 的知识，便于后续对每个 *PTX* 代码的转码；
+ 对于前面遇到的 `error: illegal SGPR to VGPR copy` 问题的原因在于 `faiss` 库中使用的 `PTX` 代码，在 *HIP* 中不支持。

## 2023/6/28

+ 目前已经完成 `faiss` 库中所有 *PTX* 到 HIP 代码的实现，并通过了 `faiss` 库的编译；

+ 在进行 `faiss` 库测试时，169 个测试案例，就有 81 个测试测试不通过，下面是测试不通过的部分信息：

  ```bash
  Running tests...
  Test project /home/dengww/Ports/faiss/build
          Start   1: BinaryFlat.accuracy
    1/169 Test   #1: BinaryFlat.accuracy .....................................   Passed    0.15 sec
          Start   2: TestIvlistDealloc.IVFFlat
    2/169 Test   #2: TestIvlistDealloc.IVFFlat ...............................   Passed    0.78 sec
          Start   3: TestIvlistDealloc.IVFSQ
    3/169 Test   #3: TestIvlistDealloc.IVFSQ .................................   Passed    0.45 sec
          Start   4: TestIvlistDealloc.IVFPQ
    4/169 Test   #4: TestIvlistDealloc.IVFPQ .................................   Passed    0.86 sec
          Start   5: IVFPQ.codec
    5/169 Test   #5: IVFPQ.codec .............................................   Passed    1.92 sec
          Start   6: IVFPQ.accuracy
    6/169 Test   #6: IVFPQ.accuracy ..........................................   Passed    0.64 sec
          Start   7: TestLowLevelIVF.IVFFlatL2
    7/169 Test   #7: TestLowLevelIVF.IVFFlatL2 ...............................   Passed    0.13 sec
          Start   8: TestLowLevelIVF.PCAIVFFlatL2
    8/169 Test   #8: TestLowLevelIVF.PCAIVFFlatL2 ............................   Passed    0.16 sec
          Start   9: TestLowLevelIVF.IVFFlatIP
    9/169 Test   #9: TestLowLevelIVF.IVFFlatIP ...............................   Passed    0.12 sec
          Start  10: TestLowLevelIVF.IVFSQL2
   10/169 Test  #10: TestLowLevelIVF.IVFSQL2 .................................   Passed    0.20 sec
          Start  11: TestLowLevelIVF.IVFSQIP
   11/169 Test  #11: TestLowLevelIVF.IVFSQIP .................................   Passed    0.17 sec
          Start  12: TestLowLevelIVF.IVFPQL2
   12/169 Test  #12: TestLowLevelIVF.IVFPQL2 .................................   Passed    0.49 sec
          Start  13: TestLowLevelIVF.IVFPQIP
   13/169 Test  #13: TestLowLevelIVF.IVFPQIP .................................   Passed    0.57 sec
          Start  14: TestLowLevelIVF.IVFBinary
   14/169 Test  #14: TestLowLevelIVF.IVFBinary ...............................   Passed    0.16 sec
          Start  15: TestLowLevelIVF.ThreadedSearch
   15/169 Test  #15: TestLowLevelIVF.ThreadedSearch ..........................   Passed    0.13 sec
          Start  16: MERGE.merge_flat_no_ids
   16/169 Test  #16: MERGE.merge_flat_no_ids .................................   Passed    0.28 sec
          Start  17: MERGE.merge_flat
   17/169 Test  #17: MERGE.merge_flat ........................................   Passed    0.13 sec
          Start  18: MERGE.merge_flat_vt
   18/169 Test  #18: MERGE.merge_flat_vt .....................................   Passed    0.11 sec
          Start  19: MERGE.merge_flat_ondisk
   19/169 Test  #19: MERGE.merge_flat_ondisk .................................   Passed    0.16 sec
          Start  20: MERGE.merge_flat_ondisk_2
   20/169 Test  #20: MERGE.merge_flat_ondisk_2 ...............................   Passed    0.12 sec
          Start  21: Threading.openmp
   21/169 Test  #21: Threading.openmp ........................................   Passed    0.26 sec
          Start  22: ONDISK.make_invlists
   22/169 Test  #22: ONDISK.make_invlists ....................................   Passed    0.39 sec
          Start  23: ONDISK.test_add
   23/169 Test  #23: ONDISK.test_add .........................................   Passed    0.17 sec
          Start  24: ONDISK.make_invlists_threaded
   24/169 Test  #24: ONDISK.make_invlists_threaded ...........................   Passed    1.37 sec
          Start  25: test_search_centroid.IVFFlat
   25/169 Test  #25: test_search_centroid.IVFFlat ............................   Passed    0.15 sec
          Start  26: test_search_centroid.PCAIVFFlat
   26/169 Test  #26: test_search_centroid.PCAIVFFlat .........................   Passed    0.18 sec
          Start  27: test_search_and_return_centroids.IVFFlat
   27/169 Test  #27: test_search_and_return_centroids.IVFFlat ................   Passed    0.16 sec
          Start  28: test_search_and_return_centroids.PCAIVFFlat
   28/169 Test  #28: test_search_and_return_centroids.PCAIVFFlat .............   Passed    0.18 sec
          Start  29: TPO.IVFFlat
   29/169 Test  #29: TPO.IVFFlat .............................................   Passed    0.14 sec
          Start  30: TPO.IVFPQ
   30/169 Test  #30: TPO.IVFPQ ...............................................   Passed    0.53 sec
          Start  31: TPO.IVFSQ
   31/169 Test  #31: TPO.IVFSQ ...............................................   Passed    0.14 sec
          Start  32: TPO.IVFFlatPP
   32/169 Test  #32: TPO.IVFFlatPP ...........................................   Passed    0.13 sec
          Start  33: TSEL.IVFFlat
   33/169 Test  #33: TSEL.IVFFlat ............................................   Passed    0.11 sec
          Start  34: TSEL.IVFFPQ
   34/169 Test  #34: TSEL.IVFFPQ .............................................   Passed    0.29 sec
          Start  35: TSEL.IVFFSQ
   35/169 Test  #35: TSEL.IVFFSQ .............................................   Passed    0.05 sec
          Start  36: TPOB.IVF
   36/169 Test  #36: TPOB.IVF ................................................   Passed    0.08 sec
          Start  37: PQEncoderGeneric.encode
   37/169 Test  #37: PQEncoderGeneric.encode .................................   Passed    0.10 sec
          Start  38: PQEncoder8.encode
   38/169 Test  #38: PQEncoder8.encode .......................................   Passed    0.10 sec
          Start  39: PQEncoder16.encode
   39/169 Test  #39: PQEncoder16.encode ......................................   Passed    0.13 sec
          Start  40: PQFastScan.set_packed_element
   40/169 Test  #40: PQFastScan.set_packed_element ...........................   Passed    0.35 sec
          Start  41: SlidingWindow.IVFFlat
   41/169 Test  #41: SlidingWindow.IVFFlat ...................................   Passed    2.79 sec
          Start  42: SlidingWindow.PCAIVFFlat
   42/169 Test  #42: SlidingWindow.PCAIVFFlat ................................   Passed    2.86 sec
          Start  43: SlidingInvlists.IVFFlat
   43/169 Test  #43: SlidingInvlists.IVFFlat .................................   Passed    2.80 sec
          Start  44: SlidingInvlists.PCAIVFFlat
   44/169 Test  #44: SlidingInvlists.PCAIVFFlat ..............................   Passed    2.91 sec
          Start  45: ThreadedIndex.SingleException
   45/169 Test  #45: ThreadedIndex.SingleException ...........................   Passed    1.12 sec
          Start  46: ThreadedIndex.MultipleException
   46/169 Test  #46: ThreadedIndex.MultipleException .........................   Passed    1.07 sec
          Start  47: ThreadedIndex.TestReplica
   47/169 Test  #47: ThreadedIndex.TestReplica ...............................   Passed    0.12 sec
          Start  48: ThreadedIndex.TestShards
   48/169 Test  #48: ThreadedIndex.TestShards ................................   Passed    0.13 sec
          Start  49: TRANS.IVFFlat
   49/169 Test  #49: TRANS.IVFFlat ...........................................   Passed    0.12 sec
          Start  50: TRANS.IVFFlatPreproc
   50/169 Test  #50: TRANS.IVFFlatPreproc ....................................   Passed    0.09 sec
          Start  51: MEM_LEAK.ivfflat
   51/169 Test  #51: MEM_LEAK.ivfflat ........................................   Passed    8.09 sec
          Start  52: TEST_CPPCONTRIB_SA_DECODE.D256_IVF256_PQ16
   52/169 Test  #52: TEST_CPPCONTRIB_SA_DECODE.D256_IVF256_PQ16 ..............   Passed    1.59 sec
          Start  53: TEST_CPPCONTRIB_SA_DECODE.D256_IVF256_PQ8
   53/169 Test  #53: TEST_CPPCONTRIB_SA_DECODE.D256_IVF256_PQ8 ...............   Passed    0.90 sec
          Start  54: TEST_CPPCONTRIB_SA_DECODE.D192_IVF256_PQ24
   54/169 Test  #54: TEST_CPPCONTRIB_SA_DECODE.D192_IVF256_PQ24 ..............   Passed    2.02 sec
          Start  55: TEST_CPPCONTRIB_SA_DECODE.D192_IVF256_PQ16
   55/169 Test  #55: TEST_CPPCONTRIB_SA_DECODE.D192_IVF256_PQ16 ..............   Passed    1.93 sec
          Start  56: TEST_CPPCONTRIB_SA_DECODE.D192_IVF256_PQ12
   56/169 Test  #56: TEST_CPPCONTRIB_SA_DECODE.D192_IVF256_PQ12 ..............   Passed    1.44 sec
          Start  57: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ40
   57/169 Test  #57: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ40 ..............   Passed    3.16 sec
          Start  58: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ20
   58/169 Test  #58: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ20 ..............   Passed    1.64 sec
          Start  59: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ10
   59/169 Test  #59: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ10 ..............   Passed    1.32 sec
          Start  60: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ8
   60/169 Test  #60: TEST_CPPCONTRIB_SA_DECODE.D160_IVF256_PQ8 ...............   Passed    1.13 sec
          Start  61: TEST_CPPCONTRIB_SA_DECODE.D128_IVF256_PQ8
   61/169 Test  #61: TEST_CPPCONTRIB_SA_DECODE.D128_IVF256_PQ8 ...............   Passed    1.04 sec
          Start  62: TEST_CPPCONTRIB_SA_DECODE.D128_IVF256_PQ4
   62/169 Test  #62: TEST_CPPCONTRIB_SA_DECODE.D128_IVF256_PQ4 ...............   Passed    1.15 sec
          Start  63: TEST_CPPCONTRIB_SA_DECODE.D64_IVF256_PQ16
   63/169 Test  #63: TEST_CPPCONTRIB_SA_DECODE.D64_IVF256_PQ16 ...............   Passed    0.94 sec
          Start  64: TEST_CPPCONTRIB_SA_DECODE.D64_IVF256_PQ8
   64/169 Test  #64: TEST_CPPCONTRIB_SA_DECODE.D64_IVF256_PQ8 ................   Passed    1.11 sec
          Start  65: TEST_CPPCONTRIB_SA_DECODE.D256_Residual4x8_PQ16
   65/169 Test  #65: TEST_CPPCONTRIB_SA_DECODE.D256_Residual4x8_PQ16 .........   Passed    1.79 sec
          Start  66: TEST_CPPCONTRIB_SA_DECODE.D256_Residual4x8_PQ8
   66/169 Test  #66: TEST_CPPCONTRIB_SA_DECODE.D256_Residual4x8_PQ8 ..........   Passed    1.09 sec
          Start  67: TEST_CPPCONTRIB_SA_DECODE.D160_Residual4x8_PQ10
   67/169 Test  #67: TEST_CPPCONTRIB_SA_DECODE.D160_Residual4x8_PQ10 .........   Passed    1.46 sec
          Start  68: TEST_CPPCONTRIB_SA_DECODE.D160_Residual2x8_PQ10
   68/169 Test  #68: TEST_CPPCONTRIB_SA_DECODE.D160_Residual2x8_PQ10 .........   Passed    1.55 sec
          Start  69: TEST_CPPCONTRIB_SA_DECODE.D160_Residual1x8_PQ10
   69/169 Test  #69: TEST_CPPCONTRIB_SA_DECODE.D160_Residual1x8_PQ10 .........   Passed    1.14 sec
          Start  70: TEST_CPPCONTRIB_SA_DECODE.D128_Residual4x8_PQ8
   70/169 Test  #70: TEST_CPPCONTRIB_SA_DECODE.D128_Residual4x8_PQ8 ..........   Passed    1.47 sec
          Start  71: TEST_CPPCONTRIB_SA_DECODE.D128_Residual4x8_PQ4
   71/169 Test  #71: TEST_CPPCONTRIB_SA_DECODE.D128_Residual4x8_PQ4 ..........   Passed    1.02 sec
          Start  72: TEST_CPPCONTRIB_SA_DECODE.D64_Residual4x8_PQ8
   72/169 Test  #72: TEST_CPPCONTRIB_SA_DECODE.D64_Residual4x8_PQ8 ...........   Passed    1.51 sec
          Start  73: TEST_CPPCONTRIB_SA_DECODE.D64_Residual4x8_PQ4
   73/169 Test  #73: TEST_CPPCONTRIB_SA_DECODE.D64_Residual4x8_PQ4 ...........   Passed    1.25 sec
          Start  74: TEST_CPPCONTRIB_SA_DECODE.D256_IVF1024_PQ16
   74/169 Test  #74: TEST_CPPCONTRIB_SA_DECODE.D256_IVF1024_PQ16 .............   Passed    1.86 sec
          Start  75: TEST_CPPCONTRIB_SA_DECODE.D64_Residual1x9_PQ8
   75/169 Test  #75: TEST_CPPCONTRIB_SA_DECODE.D64_Residual1x9_PQ8 ...........   Passed    0.75 sec
          Start  76: TEST_CPPCONTRIB_SA_DECODE.D256_PQ16
   76/169 Test  #76: TEST_CPPCONTRIB_SA_DECODE.D256_PQ16 .....................   Passed    1.66 sec
          Start  77: TEST_CPPCONTRIB_SA_DECODE.D160_PQ20
   77/169 Test  #77: TEST_CPPCONTRIB_SA_DECODE.D160_PQ20 .....................   Passed    1.29 sec
          Start  78: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAXFP16_IVF256_PQ16
   78/169 Test  #78: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAXFP16_IVF256_PQ16 ...   Passed    1.86 sec
          Start  79: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAXFP16_PQ16
   79/169 Test  #79: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAXFP16_PQ16 ..........   Passed    1.31 sec
          Start  80: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAX_IVF256_PQ16
   80/169 Test  #80: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAX_IVF256_PQ16 .......   Passed    2.10 sec
          Start  81: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAX_PQ16
   81/169 Test  #81: TEST_CPPCONTRIB_SA_DECODE.D256_MINMAX_PQ16 ..............   Passed    1.61 sec
          Start  82: TEST_CPPCONTRIB_UINTREADER.Test8bit
   82/169 Test  #82: TEST_CPPCONTRIB_UINTREADER.Test8bit .....................   Passed    0.10 sec
          Start  83: TEST_CPPCONTRIB_UINTREADER.Test10bit
   83/169 Test  #83: TEST_CPPCONTRIB_UINTREADER.Test10bit ....................   Passed    0.11 sec
          Start  84: TEST_CPPCONTRIB_UINTREADER.Test16bit
   84/169 Test  #84: TEST_CPPCONTRIB_UINTREADER.Test16bit ....................   Passed    0.05 sec
          Start  85: TestCodePacking.NonInterleavedCodes_UnpackPack
   85/169 Test  #85: TestCodePacking.NonInterleavedCodes_UnpackPack ..........   Passed    0.08 sec
          Start  86: TestCodePacking.NonInterleavedCodes_PackUnpack
   86/169 Test  #86: TestCodePacking.NonInterleavedCodes_PackUnpack ..........   Passed    0.07 sec
          Start  87: TestCodePacking.InterleavedCodes_UnpackPack
   87/169 Test  #87: TestCodePacking.InterleavedCodes_UnpackPack .............   Passed    0.16 sec
          Start  88: TestCodePacking.InterleavedCodes_PackUnpack
   88/169 Test  #88: TestCodePacking.InterleavedCodes_PackUnpack .............   Passed    0.14 sec
          Start  89: TestGpuIndexFlat.IP_Float32
   89/169 Test  #89: TestGpuIndexFlat.IP_Float32 .............................Subprocess aborted***Exception:   0.15 sec
          Start  90: TestGpuIndexFlat.L1_Float32
   90/169 Test  #90: TestGpuIndexFlat.L1_Float32 .............................Subprocess aborted***Exception:   0.10 sec
          Start  91: TestGpuIndexFlat.Lp_Float32
   91/169 Test  #91: TestGpuIndexFlat.Lp_Float32 .............................Subprocess aborted***Exception:   0.10 sec
          Start  92: TestGpuIndexFlat.L2_Float32
   92/169 Test  #92: TestGpuIndexFlat.L2_Float32 .............................Subprocess aborted***Exception:   0.10 sec
          Start  93: TestGpuIndexFlat.L2_Float32_K1
   93/169 Test  #93: TestGpuIndexFlat.L2_Float32_K1 ..........................Subprocess aborted***Exception:   0.11 sec
          Start  94: TestGpuIndexFlat.IP_Float16
   94/169 Test  #94: TestGpuIndexFlat.IP_Float16 .............................Subprocess aborted***Exception:   0.10 sec
          Start  95: TestGpuIndexFlat.L2_Float16
   95/169 Test  #95: TestGpuIndexFlat.L2_Float16 .............................Subprocess aborted***Exception:   0.10 sec
          Start  96: TestGpuIndexFlat.L2_Float16_K1
   96/169 Test  #96: TestGpuIndexFlat.L2_Float16_K1 ..........................Subprocess aborted***Exception:   0.11 sec
          Start  97: TestGpuIndexFlat.L2_Tiling
   97/169 Test  #97: TestGpuIndexFlat.L2_Tiling ..............................Subprocess aborted***Exception:   0.10 sec
          Start  98: TestGpuIndexFlat.QueryEmpty
   98/169 Test  #98: TestGpuIndexFlat.QueryEmpty .............................Subprocess aborted***Exception:   0.11 sec
          Start  99: TestGpuIndexFlat.CopyFrom
   99/169 Test  #99: TestGpuIndexFlat.CopyFrom ...............................Subprocess aborted***Exception:   0.18 sec
          Start 100: TestGpuIndexFlat.CopyTo
  100/169 Test #100: TestGpuIndexFlat.CopyTo .................................Subprocess aborted***Exception:   0.16 sec
          Start 101: TestGpuIndexFlat.UnifiedMemory
  101/169 Test #101: TestGpuIndexFlat.UnifiedMemory ..........................Subprocess aborted***Exception:   0.11 sec
          Start 102: TestGpuIndexFlat.LargeIndex
  102/169 Test #102: TestGpuIndexFlat.LargeIndex .............................Subprocess aborted***Exception:   1.73 sec
          Start 103: TestGpuIndexFlat.Residual
  103/169 Test #103: TestGpuIndexFlat.Residual ...............................Subprocess aborted***Exception:   0.12 sec
          Start 104: TestGpuIndexFlat.Reconstruct
  104/169 Test #104: TestGpuIndexFlat.Reconstruct ............................Subprocess aborted***Exception:   0.18 sec
          Start 105: TestGpuIndexFlat.SearchAndReconstruct
  105/169 Test #105: TestGpuIndexFlat.SearchAndReconstruct ...................Subprocess aborted***Exception:   0.18 sec
          Start 106: TestGpuIndexIVFFlat.Float32_32_Add_L2
  106/169 Test #106: TestGpuIndexIVFFlat.Float32_32_Add_L2 ...................Subprocess aborted***Exception:   0.39 sec
          Start 107: TestGpuIndexIVFFlat.Float32_32_Add_IP
  107/169 Test #107: TestGpuIndexIVFFlat.Float32_32_Add_IP ...................Subprocess aborted***Exception:   0.20 sec
          Start 108: TestGpuIndexIVFFlat.Float16_32_Add_L2
  108/169 Test #108: TestGpuIndexIVFFlat.Float16_32_Add_L2 ...................Subprocess aborted***Exception:   0.18 sec
          Start 109: TestGpuIndexIVFFlat.Float16_32_Add_IP
  109/169 Test #109: TestGpuIndexIVFFlat.Float16_32_Add_IP ...................Subprocess aborted***Exception:   0.21 sec
          Start 110: TestGpuIndexIVFFlat.Float32_Query_L2
  110/169 Test #110: TestGpuIndexIVFFlat.Float32_Query_L2 ....................Subprocess aborted***Exception:   0.23 sec
          Start 111: TestGpuIndexIVFFlat.Float32_Query_IP
  111/169 Test #111: TestGpuIndexIVFFlat.Float32_Query_IP ....................Subprocess aborted***Exception:   0.19 sec
          Start 112: TestGpuIndexIVFFlat.Float16_32_Query_L2
  112/169 Test #112: TestGpuIndexIVFFlat.Float16_32_Query_L2 .................Subprocess aborted***Exception:   0.20 sec
          Start 113: TestGpuIndexIVFFlat.Float16_32_Query_IP
  113/169 Test #113: TestGpuIndexIVFFlat.Float16_32_Query_IP .................Subprocess aborted***Exception:   0.19 sec
          Start 114: TestGpuIndexIVFFlat.Float32_Query_L2_64
  114/169 Test #114: TestGpuIndexIVFFlat.Float32_Query_L2_64 .................Subprocess aborted***Exception:   0.19 sec
          Start 115: TestGpuIndexIVFFlat.Float32_Query_IP_64
  115/169 Test #115: TestGpuIndexIVFFlat.Float32_Query_IP_64 .................Subprocess aborted***Exception:   0.25 sec
          Start 116: TestGpuIndexIVFFlat.Float32_Query_L2_128
  116/169 Test #116: TestGpuIndexIVFFlat.Float32_Query_L2_128 ................Subprocess aborted***Exception:   0.20 sec
          Start 117: TestGpuIndexIVFFlat.Float32_Query_IP_128
  117/169 Test #117: TestGpuIndexIVFFlat.Float32_Query_IP_128 ................Subprocess aborted***Exception:   0.20 sec
          Start 118: TestGpuIndexIVFFlat.Float32_32_CopyTo
  118/169 Test #118: TestGpuIndexIVFFlat.Float32_32_CopyTo ...................Subprocess aborted***Exception:   0.16 sec
          Start 119: TestGpuIndexIVFFlat.Float32_32_CopyFrom
  119/169 Test #119: TestGpuIndexIVFFlat.Float32_32_CopyFrom .................Subprocess aborted***Exception:   0.22 sec
          Start 120: TestGpuIndexIVFFlat.Float32_negative
  120/169 Test #120: TestGpuIndexIVFFlat.Float32_negative ....................Subprocess aborted***Exception:   0.16 sec
          Start 121: TestGpuIndexIVFFlat.QueryNaN
  121/169 Test #121: TestGpuIndexIVFFlat.QueryNaN ............................Subprocess aborted***Exception:   0.18 sec
          Start 122: TestGpuIndexIVFFlat.AddNaN
  122/169 Test #122: TestGpuIndexIVFFlat.AddNaN ..............................Subprocess aborted***Exception:   0.10 sec
          Start 123: TestGpuIndexIVFFlat.UnifiedMemory
  123/169 Test #123: TestGpuIndexIVFFlat.UnifiedMemory .......................Subprocess aborted***Exception:   0.37 sec
          Start 124: TestGpuIndexBinaryFlat.Test8
  124/169 Test #124: TestGpuIndexBinaryFlat.Test8 ............................Subprocess aborted***Exception:   0.10 sec
          Start 125: TestGpuIndexBinaryFlat.Test32
  125/169 Test #125: TestGpuIndexBinaryFlat.Test32 ...........................Subprocess aborted***Exception:   0.11 sec
          Start 126: TestGpuMemoryException.AddException
  126/169 Test #126: TestGpuMemoryException.AddException .....................Subprocess aborted***Exception:   0.10 sec
          Start 127: TestGpuIndexIVFPQ.Query_L2
  127/169 Test #127: TestGpuIndexIVFPQ.Query_L2 ..............................Subprocess aborted***Exception:   0.53 sec
          Start 128: TestGpuIndexIVFPQ.Query_L2_MMCodeDistance
  128/169 Test #128: TestGpuIndexIVFPQ.Query_L2_MMCodeDistance ...............Subprocess aborted***Exception:   0.76 sec
          Start 129: TestGpuIndexIVFPQ.Query_IP_MMCodeDistance
  129/169 Test #129: TestGpuIndexIVFPQ.Query_IP_MMCodeDistance ...............Subprocess aborted***Exception:   0.92 sec
          Start 130: TestGpuIndexIVFPQ.Query_IP
  130/169 Test #130: TestGpuIndexIVFPQ.Query_IP ..............................Subprocess aborted***Exception:   0.55 sec
          Start 131: TestGpuIndexIVFPQ.Float16Coarse
  131/169 Test #131: TestGpuIndexIVFPQ.Float16Coarse .........................Subprocess aborted***Exception:   0.59 sec
          Start 132: TestGpuIndexIVFPQ.Add_L2
  132/169 Test #132: TestGpuIndexIVFPQ.Add_L2 ................................Subprocess aborted***Exception:   0.84 sec
          Start 133: TestGpuIndexIVFPQ.Add_IP
  133/169 Test #133: TestGpuIndexIVFPQ.Add_IP ................................Subprocess aborted***Exception:   0.49 sec
          Start 134: TestGpuIndexIVFPQ.CopyTo
  134/169 Test #134: TestGpuIndexIVFPQ.CopyTo ................................Subprocess aborted***Exception:   0.12 sec
          Start 135: TestGpuIndexIVFPQ.CopyFrom
  135/169 Test #135: TestGpuIndexIVFPQ.CopyFrom ..............................Subprocess aborted***Exception:   0.55 sec
          Start 136: TestGpuIndexIVFPQ.QueryNaN
  136/169 Test #136: TestGpuIndexIVFPQ.QueryNaN ..............................Subprocess aborted***Exception:   0.18 sec
          Start 137: TestGpuIndexIVFPQ.AddNaN
  137/169 Test #137: TestGpuIndexIVFPQ.AddNaN ................................Subprocess aborted***Exception:   0.10 sec
          Start 138: TestGpuIndexIVFPQ.UnifiedMemory
  138/169 Test #138: TestGpuIndexIVFPQ.UnifiedMemory .........................Subprocess aborted***Exception:   1.84 sec
          Start 139: TestGpuIndexIVFScalarQuantizer.CopyTo_fp16
  139/169 Test #139: TestGpuIndexIVFScalarQuantizer.CopyTo_fp16 ..............Subprocess aborted***Exception:   0.17 sec
          Start 140: TestGpuIndexIVFScalarQuantizer.CopyTo_8bit
  140/169 Test #140: TestGpuIndexIVFScalarQuantizer.CopyTo_8bit ..............Subprocess aborted***Exception:   0.19 sec
          Start 141: TestGpuIndexIVFScalarQuantizer.CopyTo_8bit_uniform
  141/169 Test #141: TestGpuIndexIVFScalarQuantizer.CopyTo_8bit_uniform ......Subprocess aborted***Exception:   0.18 sec
          Start 142: TestGpuIndexIVFScalarQuantizer.CopyTo_6bit
  142/169 Test #142: TestGpuIndexIVFScalarQuantizer.CopyTo_6bit ..............Subprocess aborted***Exception:   0.17 sec
          Start 143: TestGpuIndexIVFScalarQuantizer.CopyTo_4bit
  143/169 Test #143: TestGpuIndexIVFScalarQuantizer.CopyTo_4bit ..............Subprocess aborted***Exception:   0.17 sec
          Start 144: TestGpuIndexIVFScalarQuantizer.CopyTo_4bit_uniform
  144/169 Test #144: TestGpuIndexIVFScalarQuantizer.CopyTo_4bit_uniform ......Subprocess aborted***Exception:   0.14 sec
          Start 145: TestGpuIndexIVFScalarQuantizer.CopyFrom_fp16
  145/169 Test #145: TestGpuIndexIVFScalarQuantizer.CopyFrom_fp16 ............Subprocess aborted***Exception:   0.23 sec
          Start 146: TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit
  146/169 Test #146: TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit ............Subprocess aborted***Exception:   0.34 sec
          Start 147: TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit_uniform
  147/169 Test #147: TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit_uniform ....Subprocess aborted***Exception:   0.19 sec
          Start 148: TestGpuIndexIVFScalarQuantizer.CopyFrom_6bit
  148/169 Test #148: TestGpuIndexIVFScalarQuantizer.CopyFrom_6bit ............Subprocess aborted***Exception:   0.21 sec
          Start 149: TestGpuIndexIVFScalarQuantizer.CopyFrom_4bit
  149/169 Test #149: TestGpuIndexIVFScalarQuantizer.CopyFrom_4bit ............Subprocess aborted***Exception:   0.16 sec
          Start 150: TestGpuIndexIVFScalarQuantizer.CopyFrom_4bit_uniform
  150/169 Test #150: TestGpuIndexIVFScalarQuantizer.CopyFrom_4bit_uniform ....Subprocess aborted***Exception:   0.21 sec
          Start 151: TestGpuDistance.Transposition_RR
  151/169 Test #151: TestGpuDistance.Transposition_RR ........................Subprocess aborted***Exception:   0.18 sec
          Start 152: TestGpuDistance.Transposition_RC
  152/169 Test #152: TestGpuDistance.Transposition_RC ........................Subprocess aborted***Exception:   0.20 sec
          Start 153: TestGpuDistance.Transposition_CR
  153/169 Test #153: TestGpuDistance.Transposition_CR ........................Subprocess aborted***Exception:   0.24 sec
          Start 154: TestGpuDistance.Transposition_CC
  154/169 Test #154: TestGpuDistance.Transposition_CC ........................Subprocess aborted***Exception:   0.21 sec
          Start 155: TestGpuDistance.L1
  155/169 Test #155: TestGpuDistance.L1 ......................................Subprocess aborted***Exception:   0.22 sec
          Start 156: TestGpuDistance.L1_RC
  156/169 Test #156: TestGpuDistance.L1_RC ...................................Subprocess aborted***Exception:   0.24 sec
          Start 157: TestGpuDistance.L1_CR
  157/169 Test #157: TestGpuDistance.L1_CR ...................................Subprocess aborted***Exception:   0.21 sec
          Start 158: TestGpuDistance.L1_CC
  158/169 Test #158: TestGpuDistance.L1_CC ...................................Subprocess aborted***Exception:   0.21 sec
          Start 159: TestGpuDistance.Linf
  159/169 Test #159: TestGpuDistance.Linf ....................................Subprocess aborted***Exception:   0.31 sec
          Start 160: TestGpuDistance.Lp
  160/169 Test #160: TestGpuDistance.Lp ......................................Subprocess aborted***Exception:   0.59 sec
          Start 161: TestGpuDistance.Canberra
  161/169 Test #161: TestGpuDistance.Canberra ................................Subprocess aborted***Exception:   0.29 sec
          Start 162: TestGpuDistance.BrayCurtis
  162/169 Test #162: TestGpuDistance.BrayCurtis ..............................Subprocess aborted***Exception:   0.26 sec
          Start 163: TestGpuDistance.JensenShannon
  163/169 Test #163: TestGpuDistance.JensenShannon ...........................Subprocess aborted***Exception:   2.41 sec
          Start 164: TestGpuSelect.test
  164/169 Test #164: TestGpuSelect.test ......................................Subprocess aborted***Exception:   0.25 sec
          Start 165: TestGpuSelect.test1
  165/169 Test #165: TestGpuSelect.test1 .....................................Subprocess aborted***Exception:   0.21 sec
          Start 166: TestGpuSelect.testExact
  166/169 Test #166: TestGpuSelect.testExact .................................Subprocess aborted***Exception:   0.19 sec
          Start 167: TestGpuSelect.testWarp
  167/169 Test #167: TestGpuSelect.testWarp ..................................Subprocess aborted***Exception:   0.23 sec
          Start 168: TestGpuSelect.test1Warp
  168/169 Test #168: TestGpuSelect.test1Warp .................................Subprocess aborted***Exception:   0.21 sec
          Start 169: TestGpuSelect.testExactWarp
  169/169 Test #169: TestGpuSelect.testExactWarp .............................Subprocess aborted***Exception:   0.20 sec
  
  52% tests passed, 81 tests failed out of 169
  
  Total Test time (real) = 104.27 sec
  ```

  上述测试不通过的代码均是与 *GPU* 相关的代码，因此，可以确定的是转 *CUDA* 和 *CUDA PTX* 代码过程中，*CUDA* 和 *HIP* 在某些特性上面的不同，导致编译生成的代码在执行时产生不同的结果。

  后续的任务便是针对每个测试不通过的案例，进行相依 *GPU* 源码的修改，使其能够正确测试通过。

+ 测试不通过的主要原因有以下：

  + 矩阵乘法过程中的 `Tensor<>` 的 `getStride()` 方法获取某个维度上的步长出错；
  + 矩阵乘法结果的精度不足，这里应该就是矩阵乘法结果错误，里面应该是那里代码转码后的问题；
  + 

## 2023/6/29

+ 进一步测试和对比发现，因为在 CUDA 中的 `warpSize = 32`，而 HIP 中的 `wavefront=64` ，故而在进行如 `__shfl()`, `__shfl_xor`, `__any` 操作时，就会在计算正确性上不能保证。要解决这一问题的思路就是需要对相应的 device 代码进行适配到 `warpSize=64` 的情况。

## 2023/7/3

+ 测试失败结果：1、内存拷贝时的内存出错；2、计算结果精度不准(这可能是那部分代码出错)；3、某些 `assert` 条件不满足；

+ `faiss/gpu/impl/PQCodeLoad.h` 文件中的 *CUDA PTX* 调试，测试使用的案例为 `TestGpuIndexIVFPQ.cpp`，到 `LoadCode32<>::load()` 接口的上下文路径为：

  ```bash
  #0  faiss::gpu::runMultiPassTile (res=0x3082f80, queries=..., precompTerm1=..., precompTerm2=..., precompTerm3=..., ivfListIds=..., useFloat16Lookup=true, interleavedCodeLayout=<optimized out>, 
      bitsPerSubQuantizer=8, numSubQuantizers=8, numSubQuantizerCodes=256, listCodes=..., listIndices=..., indicesOptions=faiss::gpu::INDICES_32_BIT, listLengths=..., thrustMem=..., prefixSumOffsets=..., 
      allDistances=..., heapDistances=..., heapIndices=..., k=7, outDistances=..., outIndices=..., stream=0x30ddb10) at /home/dengww/Ports/faiss/faiss/gpu/impl/PQScanMultiPassPrecomputed.cpp:335
  #1  0x00000000004ed407 in faiss::gpu::runPQScanMultiPassPrecomputed (queries=..., precompTerm1=..., precompTerm2=..., precompTerm3=..., ivfListIds=..., useFloat16Lookup=<optimized out>, 
      interleavedCodeLayout=<optimized out>, bitsPerSubQuantizer=<optimized out>, numSubQuantizers=<optimized out>, numSubQuantizerCodes=<optimized out>, listCodes=..., listIndices=..., 
      indicesOptions=<optimized out>, listLengths=..., maxListLength=<optimized out>, k=<optimized out>, outDistances=..., outIndices=..., res=<optimized out>)
      at /home/dengww/Ports/faiss/faiss/gpu/impl/PQScanMultiPassPrecomputed.cpp:696
  #2  0x00000000004c65d7 in faiss::gpu::IVFPQ::runPQPrecomputedCodes_ (this=this@entry=0x3372470, queries=..., coarseDistances=..., coarseIndices=..., k=k@entry=7, outDistances=..., outIndices=...)
      at /home/dengww/Ports/faiss/faiss/gpu/impl/IVFPQ.cpp:683
  #3  0x00000000004c5f26 in faiss::gpu::IVFPQ::searchImpl_ (this=this@entry=0x3372470, queries=..., coarseDistances=..., coarseIndices=..., k=k@entry=7, outDistances=..., outIndices=..., 
      storePairs=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/impl/IVFPQ.cpp:580
  #4  0x00000000004c696d in faiss::gpu::IVFPQ::search (this=0x3372470, coarseQuantizer=<optimized out>, queries=..., nprobe=54, k=7, outDistances=..., outIndices=...)
      at /home/dengww/Ports/faiss/faiss/gpu/impl/IVFPQ.cpp:526
  #5  0x00000000004af8eb in faiss::gpu::GpuIndexIVF::searchImpl_ (this=0x7fffffffd978, n=7, x=0x7ffdc0000310, k=7, distances=0x7ffdc0000010, labels=<optimized out>, params=0x0)
      at /home/dengww/Ports/faiss/faiss/gpu/GpuIndexIVF.cpp:347
  #6  0x00000000004ac696 in faiss::gpu::GpuIndex::searchNonPaged_ (this=0x7fffffffd978, n=7, x=0x7bb52f0, k=7, outDistancesData=0x7ffdc0000010, outIndicesData=0x7ffdc0000110, params=<optimized out>)
      at /home/dengww/Ports/faiss/faiss/gpu/GpuIndex.cpp:314
  #7  faiss::gpu::GpuIndex::search (this=0x7fffffffd978, n=7, x=0x7bb52f0, k=7, distances=<optimized out>, labels=0x7bb6800, params=0x0) at /home/dengww/Ports/faiss/faiss/gpu/GpuIndex.cpp:276
  #8  0x00000000004131a3 in faiss::gpu::compareIndices (queryVecs=..., refIndex=..., testIndex=..., numQuery=<optimized out>, k=<optimized out>, configMsg=..., maxRelativeError=<optimized out>, 
      pctMaxDiff1=<optimized out>, pctMaxDiffN=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/test/TestUtils.cpp:110
  #9  0x0000000000414fb0 in faiss::gpu::compareIndices (refIndex=..., testIndex=..., numQuery=7, dim=<optimized out>, k=7, configMsg=..., maxRelativeError=1.12103877e-44, pctMaxDiff1=0, pctMaxDiffN=0)
      at /home/dengww/Ports/faiss/faiss/gpu/test/TestUtils.cpp:145
  #10 0x000000000040b55e in TestGpuIndexIVFPQ_Query_L2_Test::TestBody (this=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/test/TestGpuIndexIVFPQ.cpp:142
  #11 0x0000000000611346 in testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void> (method=<optimized out>, location=0x671249 "the test body", object=<optimized out>)
      at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2599
  --Type <RET> for more, q to quit, c to continue without paging--
  #12 testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void> (object=<optimized out>, method=<optimized out>, location=0x671249 "the test body")
      at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2635
  #13 0x00000000005f64b8 in testing::Test::Run (this=0x2f122c0) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2674
  #14 0x00000000005f744f in testing::TestInfo::Run (this=0x2eced70) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2853
  #15 0x00000000005f7e2f in testing::TestSuite::Run (this=0x2ecf320) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:3012
  #16 0x000000000060898f in testing::internal::UnitTestImpl::RunAllTests (this=<optimized out>) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:5870
  #17 0x0000000000612016 in testing::internal::HandleSehExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool> (method=<optimized out>, 
      location=0x671ac8 "auxiliary test code (environments or event listeners)", object=<optimized out>) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2599
  #18 testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool> (object=<optimized out>, method=<optimized out>, 
      location=0x671ac8 "auxiliary test code (environments or event listeners)") at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2635
  #19 0x000000000060817c in testing::UnitTest::Run (this=0x2c3a9b0 <testing::UnitTest::GetInstance()::instance>) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:5444
  #20 0x0000000000410116 in RUN_ALL_TESTS () at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/include/gtest/gtest.h:2293
  #21 main (argc=1, argv=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/test/TestGpuIndexIVFPQ.cpp:744
  ```

  

## 2023/7/6

```bash
The following tests FAILED:
         90 - TestGpuIndexFlat.L1_Float32 (Failed)
         91 - TestGpuIndexFlat.Lp_Float32 (Failed)
         95 - TestGpuIndexFlat.L2_Float16 (Failed)
        101 - TestGpuIndexFlat.UnifiedMemory (Failed)
        102 - TestGpuIndexFlat.LargeIndex (Failed)
        105 - TestGpuIndexFlat.SearchAndReconstruct (Subprocess aborted)
        106 - TestGpuIndexIVFFlat.Float32_32_Add_L2 (Subprocess aborted)
        107 - TestGpuIndexIVFFlat.Float32_32_Add_IP (Subprocess aborted)
        108 - TestGpuIndexIVFFlat.Float16_32_Add_L2 (Subprocess aborted)
        109 - TestGpuIndexIVFFlat.Float16_32_Add_IP (Subprocess aborted)
        110 - TestGpuIndexIVFFlat.Float32_Query_L2 (Subprocess aborted)
        111 - TestGpuIndexIVFFlat.Float32_Query_IP (Subprocess aborted)
        112 - TestGpuIndexIVFFlat.Float16_32_Query_L2 (Subprocess aborted)
        113 - TestGpuIndexIVFFlat.Float16_32_Query_IP (Subprocess aborted)
        114 - TestGpuIndexIVFFlat.Float32_Query_L2_64 (Subprocess aborted)
        115 - TestGpuIndexIVFFlat.Float32_Query_IP_64 (Failed)
        116 - TestGpuIndexIVFFlat.Float32_Query_L2_128 (Subprocess aborted)
        117 - TestGpuIndexIVFFlat.Float32_Query_IP_128 (Subprocess aborted)
        118 - TestGpuIndexIVFFlat.Float32_32_CopyTo (Subprocess aborted)
        119 - TestGpuIndexIVFFlat.Float32_32_CopyFrom (Subprocess aborted)
        120 - TestGpuIndexIVFFlat.Float32_negative (Subprocess aborted)
        121 - TestGpuIndexIVFFlat.QueryNaN (Failed)
        122 - TestGpuIndexIVFFlat.AddNaN (Subprocess aborted)
        123 - TestGpuIndexIVFFlat.UnifiedMemory (Subprocess aborted)
        127 - TestGpuIndexIVFPQ.Query_L2 (Subprocess aborted)
        128 - TestGpuIndexIVFPQ.Query_L2_MMCodeDistance (Subprocess aborted)
        129 - TestGpuIndexIVFPQ.Query_IP_MMCodeDistance (Subprocess aborted)
        130 - TestGpuIndexIVFPQ.Query_IP (Subprocess aborted)
        131 - TestGpuIndexIVFPQ.Float16Coarse (Subprocess aborted)
        132 - TestGpuIndexIVFPQ.Add_L2 (Subprocess aborted)
        133 - TestGpuIndexIVFPQ.Add_IP (Subprocess aborted)
        134 - TestGpuIndexIVFPQ.CopyTo (Subprocess aborted)
        135 - TestGpuIndexIVFPQ.CopyFrom (Subprocess aborted)
        136 - TestGpuIndexIVFPQ.QueryNaN (Failed)
        138 - TestGpuIndexIVFPQ.UnifiedMemory (Subprocess aborted)
        139 - TestGpuIndexIVFScalarQuantizer.CopyTo_fp16 (Subprocess aborted)
        140 - TestGpuIndexIVFScalarQuantizer.CopyTo_8bit (Subprocess aborted)
        141 - TestGpuIndexIVFScalarQuantizer.CopyTo_8bit_uniform (Subprocess aborted)
        142 - TestGpuIndexIVFScalarQuantizer.CopyTo_6bit (Subprocess aborted)
        143 - TestGpuIndexIVFScalarQuantizer.CopyTo_4bit (Subprocess aborted)
        144 - TestGpuIndexIVFScalarQuantizer.CopyTo_4bit_uniform (Subprocess aborted)
        145 - TestGpuIndexIVFScalarQuantizer.CopyFrom_fp16 (Subprocess aborted)
        146 - TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit (Subprocess aborted)
        147 - TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit_uniform (Subprocess aborted)
        148 - TestGpuIndexIVFScalarQuantizer.CopyFrom_6bit (Subprocess aborted)
        149 - TestGpuIndexIVFScalarQuantizer.CopyFrom_4bit (Subprocess aborted)
        150 - TestGpuIndexIVFScalarQuantizer.CopyFrom_4bit_uniform (Subprocess aborted)
        151 - TestGpuDistance.Transposition_RR (Failed)
        152 - TestGpuDistance.Transposition_RC (Failed)
        153 - TestGpuDistance.Transposition_CR (Failed)
        154 - TestGpuDistance.Transposition_CC (Failed)
        155 - TestGpuDistance.L1 (Failed)
        156 - TestGpuDistance.L1_RC (Failed)
        157 - TestGpuDistance.L1_CR (Failed)
        158 - TestGpuDistance.L1_CC (Failed)
        159 - TestGpuDistance.Linf (Failed)
        160 - TestGpuDistance.Lp (Failed)
        161 - TestGpuDistance.Canberra (Failed)
        162 - TestGpuDistance.BrayCurtis (Failed)
        163 - TestGpuDistance.JensenShannon (Failed)
```

与之前的测试结果相对比，证明修改代码还是有作用。

## 2023/7/10

发现了一种判断自己写的 *CUDA PTX* 代码的替代代码是否正确的方法，那就是在 *NVIDIA* 平台下，将所有 *CUDA PTX* 代码全部换成自己写的转码，通过编译测试来判断是否正确。

通过测试，发现下面的 *test case* 没有通过：

```bash
The following tests FAILED:
        127 - TestGpuIndexIVFPQ.Query_L2 (Failed)
        128 - TestGpuIndexIVFPQ.Query_L2_MMCodeDistance (Failed)
        129 - TestGpuIndexIVFPQ.Query_IP_MMCodeDistance (Failed)
        130 - TestGpuIndexIVFPQ.Query_IP (Failed)
        131 - TestGpuIndexIVFPQ.Float16Coarse (Failed)
        132 - TestGpuIndexIVFPQ.Add_L2 (Failed)
        133 - TestGpuIndexIVFPQ.Add_IP (Failed)
        134 - TestGpuIndexIVFPQ.CopyTo (Failed)
        135 - TestGpuIndexIVFPQ.CopyFrom (Failed)
        138 - TestGpuIndexIVFPQ.UnifiedMemory (Failed)
```

因为在 *faiss* 库中只有以下三个文件：

+ `faiss/gpu/utils/LoadStoreOperators.cuh`
+ `faiss/gpu/utils/PtxUtils.cuh`
+ `faiss/gpu/impl/PQCodeLoad.cuh`

中存在 *CUDA PTX* 代码，因此，分别针对这三个文件中的 *CUDA PTX* 代码进行**转码替换**后测试，即可**分别确定**三个文件中的 *CUDA PTX* 转码是否正确。

## 2023/7/11

在分别测试上面三个文件中 *CUDA PTX* 代码之前，首先要确保 *CUDA PTX* 代码本身在测试时是没有问题的。因此，以下是 *faiss* 在 *NVIDIA* 环境下的测试结果：

```bash
100% tests passed, 0 tests failed out of 169
```

首先，我先来排除 `faiss/gpu/utils/LoadStoreOperators.cuh` 文件中的所有 *CUDA PTX* 代码是否正确，通过对每个测试进行运行，没有通过测试的 *test case* 如下：

```bash
[  FAILED  ] TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit
[  FAILED  ] TestGpuIndexIVFScalarQuantizer.CopyFrom_8bit_uniform
[  FAILED  ] TestGpuIndexIVFPQ.Add_L2
[  FAILED  ] TestGpuIndexIVFFlat.Float32_Query_L2
[  FAILED  ] TestGpuIndexIVFFlat.Float16_32_Query_L2
```

但是使用 `make test` 命令进行全局测试时，没有通过测试的 *test case* 有：

```bash
100% tests passed, 0 tests failed out of 169
```

----

上这里看得出来，直接调用测试程序可能会测试不通过，而要在编译成功后，使用 `make test` 进行测试。

接下来，便是将 `faiss/gpu/utils/LoadStoreOperators.cuh` 和`faiss/gpu/impl/PQCodeLoad.cuh` 中的 *CUDA PTX* 代码保留，`faiss/gpu/utils/PtxUtils.cuh` 中的 *CUDA PTX* 代码替换为修改的代码，下面是使用 `make test` 运行的结果：

```bash
100% tests passed, 0 tests failed out of 169
```

## 2023/7/12

接下来，便是将 `faiss/gpu/utils/PtxUtils.cuh` 和`faiss/gpu/utils/LoadStoreOperators.cuh` 中的 *CUDA PTX* 代码保留，`faiss/gpu/impl/PQCodeLoad.cuh` 中的 *CUDA PTX* 代码替换为修改的代码，下面是使用 `make test` 运行的结果：

```bash
100% tests passed, 0 tests failed out of 169
```

从上面看，每个 *CUDA PTX* 替换为转换的码，都能正确通过测试。

接下来，将所有 *CDUA PTX* 替换为转码，看看其测试结果如何：

```bash
100% tests passed, 0 tests failed out of 169
```

从上面的测试结果来看，所有转的 *CUDA PTX* 代码是正确的。导致其他测试失败的原因只能是 *faiss* 库中对 *CUDA* 适配方面的问题了。

## 2023/7/13

现在面临一个问题，在运行程序时的内存地址越界访问问题，目前正在着手解决这个问题。

## 2023/7/14

<img src="..\..\weituinfo-documents\images\knowhere_ports\indexflat_search_results_cuda.png" style="zoom:150%;" />

<center>
    CUDA 下的 k-select 分别在CPU/GPU上的结果
</center>


<img src="..\..\weituinfo-documents\images\knowhere_ports\indexflat_search_results_hip.png" style="zoom:150%;" />

<center>
    HIP 下的 k-select 分别在CPU/GPU上的结果
</center>


可以看出，`faiss::gpu::GpuIndexFlat` 算法在进行 `k-selection` 的过程中出现了错误，导致了结果与 *CPU* 不一致。更有甚者，在 *HIP* 中，有的时候，测试结果的搜索索引值为负值，这是一个更大的错误。

因此，接下来的任务便是能够把 `faiss::gpu::GpuIndexFlat` 算法进行 *k-select* 的过程理清，调试这个计算过程，看看这个过程中，到底是哪个地方出错了。



先看看 `faiss::gpu::GpuIndexFlat` 在进行 *k-select* 时的大致调用栈情况：

```bash
#0  faiss::gpu::runDistance<float> (computeL2=true, res=0x10d8b50, res@entry=0x7fff0000000a, stream=0x111e2e0, stream@entry=0x7fffffffd4f0, centroids=..., centroidsRowMajor=true, 
    centroidNorms=centroidNorms@entry=0x1251e18, queries=..., queriesRowMajor=true, k=10, outDistances=..., outIndices=..., ignoreOutDistances=<optimized out>)
    at /home/dengww/Ports/faiss/faiss/gpu/impl/Distance.cpp:134
#1  0x00000000004e832e in faiss::gpu::runL2Distance<float> (res=0x10d8b50, res@entry=0x7fffffffd4f0, stream=0x111e201, stream@entry=0x1251d98, centroids=..., centroidsRowMajor=true, 
    centroidNorms=0x1251e18, queries=..., k=10, outDistances=..., outIndices=..., queriesRowMajor=<optimized out>, ignoreOutDistances=<optimized out>)
    at /home/dengww/Ports/faiss/faiss/gpu/impl/Distance.cpp:420
#2  faiss::gpu::runL2Distance (res=0x10d8b50, res@entry=0x7fffffffd4f0, stream=0x111e201, stream@entry=0x7fffffffd538, vectors=..., vectorsRowMajor=true, vectorNorms=vectorNorms@entry=0x1251e18, 
    queries=..., queriesRowMajor=<optimized out>, k=10, outDistances=..., outIndices=..., ignoreOutDistances=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/impl/Distance.cpp:559
#3  0x0000000000447853 in faiss::gpu::bfKnnOnDevice<float> (resources=<optimized out>, resources@entry=0x10d8b50, device=<optimized out>, stream=<optimized out>, stream@entry=0x111e2e0, 
    vectors=..., vectorsRowMajor=<optimized out>, vectorNorms=vectorNorms@entry=0x1251e18, queries=..., queriesRowMajor=true, k=<optimized out>, metric=<optimized out>, metricArg=<optimized out>, 
    metricArg@entry=0, outDistances=..., outIndices=..., ignoreOutDistances=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/impl/Distance.h:266
#4  0x0000000000445c72 in faiss::gpu::FlatIndex::query (this=<optimized out>, input=..., k=<optimized out>, metric=<optimized out>, metricArg=<optimized out>, outDistances=..., outIndices=..., 
    exactDistance=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/impl/FlatIndex.cpp:119
#5  0x000000000043c15e in faiss::gpu::GpuIndexFlat::searchImpl_ (this=0x7fffffffda98, n=10, x=<optimized out>, k=10, distances=<optimized out>, labels=0x7fff850e5000, params=<optimized out>)
    at /home/dengww/Ports/faiss/faiss/gpu/GpuIndexFlat.cpp:229
#6  0x0000000000437f46 in faiss::gpu::GpuIndex::searchNonPaged_ (this=0x7fffffffda98, n=10, x=0x111d710, k=10, outDistancesData=0x7fff850e4000, outIndicesData=0x7fff850e5000, 
    params=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/GpuIndex.cpp:314
#7  faiss::gpu::GpuIndex::search (this=0x7fffffffda98, n=10, x=0x111d710, k=10, distances=<optimized out>, labels=0x1a1f3d0, params=0x0) at /home/dengww/Ports/faiss/faiss/gpu/GpuIndex.cpp:276
#8  0x0000000000438d5e in faiss::gpu::GpuIndex::search_and_reconstruct (this=0x7fffffffda98, n=10, x=0x111e201, k=10, distances=0x1, labels=0x1a1f3d0, recons=0x1342e70, params=0x0)
    at /home/dengww/Ports/faiss/faiss/gpu/GpuIndex.cpp:292
#9  0x000000000040cf34 in TestGpuIndexFlat_SearchAndReconstruct_Test::TestBody (this=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/test/TestGpuIndexFlat.cpp:581
#10 0x0000000000549316 in testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void> (method=<optimized out>, location=0x57c4c9 "the test body", object=<optimized out>)
    at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2599
#11 testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void> (object=<optimized out>, method=<optimized out>, location=0x57c4c9 "the test body")
--Type <RET> for more, q to quit, c to continue without paging--
   _deps/googletest-src/googletest/src/gtest.cc:2635
#12 0x000000000052e488 in testing::Test::Run (this=0xff48e0) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2674
#13 0x000000000052f41f in testing::TestInfo::Run (this=0xfebe60) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2853
#14 0x000000000052fdff in testing::TestSuite::Run (this=0xfea320) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:3012
#15 0x000000000054095f in testing::internal::UnitTestImpl::RunAllTests (this=<optimized out>) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:5870
#16 0x0000000000549fe6 in testing::internal::HandleSehExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool> (method=<optimized out>, 
    location=0x57cd48 "auxiliary test code (environments or event listeners)", object=<optimized out>) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2599
#17 testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool> (object=<optimized out>, method=<optimized out>, 
    location=0x57cd48 "auxiliary test code (environments or event listeners)") at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:2635
#18 0x000000000054014c in testing::UnitTest::Run (this=0xd55558 <testing::UnitTest::GetInstance()::instance>) at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/src/gtest.cc:5444
#19 0x000000000040dbd6 in RUN_ALL_TESTS () at /home/dengww/Ports/faiss/build/_deps/googletest-src/googletest/include/gtest/gtest.h:2293
#20 main (argc=1, argv=<optimized out>) at /home/dengww/Ports/faiss/faiss/gpu/test/TestGpuIndexFlat.cpp:635
```

通过之前的测试结果，可以发现，在进行计算距离时，其结果也是不正确的。因此，得先确保向量之间的距离计算正确，先解决距离计算部分的代码，与其相对于的测试代码为 `TestGpuDistance.cpp`。

## 2023/7/17

+ 加入了 `TestL2NormKernel.cpp` 用于测试 *faiss* 库中计算 *L2* 范数和矩阵转置 *kernel* 是否出错；
+ 加入了 `TestMatrixMult.cpp` 用于测试矩阵乘法运算的 *kernel* 是否正确；

## 2023/7/18

+ 加入了 `TestL2SelectMin.cpp` 用于测试 *L2* 计算距离下的 *k-select* 最小值 的 *kernel* 是否正确，通过测试发现，*faiss* 库中对不同的 `k` 值会使用不同 *kernel* 参数，里面在 `k<=32` 时，会有适配 *CUDA* 的参数(即这里会使用 `NUM_WARP_Q=32` 的线程束队列大小，但是在 *HIP* 中，一个线程束的大小为 64，这里最小的线程束队列大小也得是 64)，在做了修改后，能够通过计算 *L2* 距离和 *k-select* 的测试 *case*；
+ 通过测试 `GeneralDistance.h` 中的 `runGeneralDistance<>` *kernel* 接口(用于计算向量一般距离的 *kernel*，包括 *L1, L2, Lp* 等距离)，发现里面有对 *CUDA* 的适配。这里主要是类似矩阵乘法运算过程，会将***矩阵乘法***计算的两个矩阵分别按行按列进行划分为一个个块，在一个线程块中计算相对应的块中小矩阵的"乘法运算"，因为这里默认使用 `kWarpSize * kWarpSize` 大小的线程块，故而，这里超出了 *DCU* 能够使用的最大线程块大小，故而导致计算错误。
+ 已通过 `TestGpuIndexFlat.cpp` 中的所有测试，因此，关于 `IndexFlat` 算法的所有操作应该是可以直接使用了；

## 2023/7/19

+ 着手阅读 `GpuIndexIVFFlat` 算法相关的代码；
+ 将 *faiss* 库中的 *CUDA-32* 硬编码部分，修改为适配 *HIP-64* 硬编码的实现后，编译测试，通过了 `GpuIndexFlat`, `GpuIndexIVFFlat`, `GpuIndexIVFPQ` 所有算法的测试；

## 2023/7/20

+ 修改了 `GpuIndexIVFScalarQuantizer` 算法中对 *CUDA-32* 的硬编码代码，使其适配 *HIP-64* 的硬编码情况。

+ 最后，运行 `make test`，通过了 *faiss* 库中的 *CPU* 侧和 *GPU* 侧的所有测试：

  ```bash
  100% tests passed, 0 tests failed out of 177
  
  Total Test time (real) = 1061.99 sec
  ```

+ 做一个整个 *faiss* 库中对 *CUDA-32* 硬编码代码的调整，将所有 32 硬编码的代码注释改为适配 *HIP* 的 64 硬编码注释；

+ 再做一次 *faiss* 库中的 `CMakeLists.txt` 构建文件的修改，以适配 *ROCm* 平台；

## 2023/7/21

+ 将移植好的 *faiss* 库加入到 *knowhere* 这个向量数据库中一起编译，编译完后进行测试：

  ```bash
  All tests passed (176 assertions in 7 test cases)
  ```

  发现能够通过所有的测试。

+ 解决在编译过程中出现的 *warnings*；
