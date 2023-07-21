# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_GPU_SRC
  GpuAutoTune.cpp
  GpuCloner.cpp
  GpuClonerOptions.cpp
  GpuDistance.cu
  GpuIcmEncoder.cu
  GpuIndex.cu
  GpuIndexBinaryFlat.cu
  GpuIndexFlat.cu
  GpuIndexIVF.cu
  GpuIndexIVFFlat.cu
  GpuIndexIVFPQ.cu
  GpuIndexIVFScalarQuantizer.cu
  # GpuIndexIVFSQHybrid.cu
  GpuResources.cpp
  StandardGpuResources.cpp
  impl/BinaryDistance.cu
  impl/BinaryFlatIndex.cu
  impl/BroadcastSum.cu
  impl/Distance.cu
  impl/FlatIndex.cu
  impl/IVFAppend.cu
  impl/IVFBase.cu
  impl/IVFFlat.cu
  impl/IVFFlatScan.cu
  impl/IVFInterleaved.cu
  impl/IVFPQ.cu
  impl/IVFUtils.cu
  impl/IVFUtilsSelect1.cu
  impl/IVFUtilsSelect2.cu
  impl/InterleavedCodes.cpp
  impl/L2Norm.cu
  impl/L2Select.cu
  impl/PQScanMultiPassPrecomputed.cu
  impl/RemapIndices.cpp
  impl/VectorResidual.cu
  impl/scan/IVFInterleaved1.cu
  # impl/scan/IVFInterleaved32.cu
  impl/scan/IVFInterleaved64.cu
  impl/scan/IVFInterleaved128.cu
  impl/scan/IVFInterleaved256.cu
  impl/scan/IVFInterleaved512.cu
  impl/scan/IVFInterleaved1024.cu
  impl/scan/IVFInterleaved2048.cu
  impl/IcmEncoder.cu
  utils/BlockSelectFloat.cu
  utils/DeviceUtils.cu
  utils/StackDeviceMemory.cpp
  utils/Timer.cpp
  utils/WarpSelectFloat.cu
  utils/blockselect/BlockSelectFloat1.cu
  # utils/blockselect/BlockSelectFloat32.cu
  utils/blockselect/BlockSelectFloat64.cu
  utils/blockselect/BlockSelectFloat128.cu
  utils/blockselect/BlockSelectFloat256.cu
  utils/blockselect/BlockSelectFloatF512.cu
  utils/blockselect/BlockSelectFloatF1024.cu
  utils/blockselect/BlockSelectFloatF2048.cu
  utils/blockselect/BlockSelectFloatT512.cu
  utils/blockselect/BlockSelectFloatT1024.cu
  utils/blockselect/BlockSelectFloatT2048.cu
  utils/warpselect/WarpSelectFloat1.cu
  utils/warpselect/WarpSelectFloat32.cu
  utils/warpselect/WarpSelectFloat64.cu
  utils/warpselect/WarpSelectFloat128.cu
  utils/warpselect/WarpSelectFloat256.cu
  utils/warpselect/WarpSelectFloatF512.cu
  utils/warpselect/WarpSelectFloatF1024.cu
  utils/warpselect/WarpSelectFloatF2048.cu
  utils/warpselect/WarpSelectFloatT512.cu
  utils/warpselect/WarpSelectFloatT1024.cu
  utils/warpselect/WarpSelectFloatT2048.cu
)

set(FAISS_GPU_HEADERS
  GpuAutoTune.h
  GpuCloner.h
  GpuClonerOptions.h
  GpuDistance.h
  GpuIcmEncoder.h
  GpuFaissAssert.h
  GpuIndex.h
  GpuIndexBinaryFlat.h
  GpuIndexFlat.h
  GpuIndexIVF.h
  GpuIndexIVFFlat.h
  GpuIndexIVFPQ.h
  GpuIndexIVFScalarQuantizer.h
  GpuIndexIVFSQHybrid.h
  GpuIndicesOptions.h
  GpuResources.h
  StandardGpuResources.h
  impl/BinaryDistance.h
  impl/BinaryFlatIndex.h
  impl/BroadcastSum.h
  impl/Distance.h
  impl/DistanceUtils.h
  impl/FlatIndex.h
  impl/GeneralDistance.h
  impl/GpuScalarQuantizer.h
  impl/IVFAppend.h
  impl/IVFBase.h
  impl/IVFFlat.h
  impl/IVFFlatScan.h
  impl/IVFInterleaved.h
  impl/IVFPQ.h
  impl/IVFUtils.h
  impl/InterleavedCodes.h
  impl/L2Norm.h
  impl/L2Select.h
  impl/PQCodeDistances-inl.h
  impl/PQCodeDistances.h
  impl/PQCodeLoad.h
  impl/PQScanMultiPassNoPrecomputed-inl.h
  impl/PQScanMultiPassNoPrecomputed.h
  impl/PQScanMultiPassPrecomputed.h
  impl/RemapIndices.h
  impl/VectorResidual.h
  impl/scan/IVFInterleavedImpl.h
  impl/IcmEncoder.h
  utils/BlockSelectImpl.h
  utils/BlockSelectKernel.h
  utils/Comparators.h
  utils/ConversionOperators.h
  utils/CopyUtils.h
  utils/DeviceDefs.h
  utils/DeviceTensor-inl.h
  utils/DeviceTensor.h
  utils/DeviceUtils.h
  utils/DeviceVector.h
  utils/Float16.h
  utils/HostTensor-inl.h
  utils/HostTensor.h
  utils/Limits.h
  utils/LoadStoreOperators.h
  utils/MathOperators.h
  utils/MatrixMult-inl.h
  utils/MatrixMult.h
  utils/MergeNetworkBlock.h
  utils/MergeNetworkUtils.h
  utils/MergeNetworkWarp.h
  utils/NoTypeTensor.h
  utils/Pair.h
  utils/PtxUtils.h
  utils/ReductionOperators.h
  utils/Reductions.h
  utils/Select.h
  utils/StackDeviceMemory.h
  utils/StaticUtils.h
  utils/Tensor-inl.h
  utils/Tensor.h
  utils/ThrustAllocator.h
  utils/Timer.h
  utils/Transpose.h
  utils/WarpPackedBits.h
  utils/WarpSelectKernel.h
  utils/WarpShuffles.h
  utils/blockselect/BlockSelectImpl.h
  utils/warpselect/WarpSelectImpl.h
)

# Export FAISS_GPU_HEADERS variable to parent scope.
set(FAISS_GPU_HEADERS ${FAISS_GPU_HEADERS} PARENT_SCOPE)

target_sources(faiss PRIVATE ${FAISS_GPU_SRC})
target_sources(faiss_avx2 PRIVATE ${FAISS_GPU_SRC})

foreach(header ${FAISS_GPU_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/gpu/${dir}
  )
endforeach()

find_package(hip REQUIRED)
find_package(rocthrust REQUIRED)
find_package(hipblas REQUIRED)
find_package(hiprand REQUIRED)
find_package(rocrand REQUIRED)
target_link_libraries(faiss PRIVATE hip::host hip::device roc::rocthrust roc::hipblas hip::hiprand roc::rocrand)
target_link_libraries(faiss_avx2 PRIVATE hip::host hip::device roc::rocthrust roc::hipblas hip::hiprand roc::rocrand)
