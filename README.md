<p>
    <img src="static/knowhere-logo.png" alt="Knowhere Logo"/>
</p>

# *knowhere* ROCm GPU backend library
本库将 [*milvus-io/knowhere*](https://github.com/milvus-io/knowhere) 从 *CUDA GPU* 后端移植到 *ROCm GPU* 后端，该库可以在 *ROCm* GPU下进行加速计算。目前，只支持 *faiss* 库，而不支持 *raft*。

关于 *ROCm GPU* 后端支持的 *knowhere* 库的编译和安装，请参考 [Kownhere README](./KNOWHERE_README.md)。

## 移植汇总
*thirdparty/faiss* 库移植适配 *ROCm-GPU* 的主要内容汇总如下：

+ *k-select* 算法中，*warpSize* 大小的线程束内的双调排序部分实现，从 CUDA-backend 的 32 warp 大小调整为 ROCm-backend 的 64 warp 大小；且在进行合并时，需要进行大小为 32 长度的合并操作(在 CUDA-backend 是 16 大小的合并操作)；
+ 对于 *k-select* 算法中，一个 warp 中会有一个 warp 队列，当 `k<=32` 时，CUDA-backend 会设置 32 的队列大小，ROCm-backend 需要设置为 64 的队列大小。这里有一个例外是 WarpSelect 实现，其 `N_WARP_Q` 大小 32 是不影响的，而对于 BlockSelect, IVFInterleavedScan, IVFInterleavedScan2 等都需要将 `N_WARP_Q` 大小设置为 64；
+ 在关于 `IVFPQ` 实现部分，CUDA-backend 需要用一个 warpSize=32 去编码 32 个向量，对应到 ROCm-backend 下，就需要用一个 warpSize=64 去编码 64 个向量；
+ 最后一部分，就是 *Warp-coalesced parallel reading and writing of packed bits*，这里需要将 CUDA-backend 的 32 个 warp 线程去读写 4-bits, 5-bits, 6-bits 的操作，改为 ROCm-backend 下 64 个 warp 线程去读写 4-bits, 5-bits, 6-bits 的操作；
+ 更多具体的移植过程，可以参考 git 的 commit 记录对边，进行具体了解；

*knowhere* 源码中和 *CUDA* 相关的代码：
+ `src/index/ivf_gpu` 目录下的 `ivf_gpu.cc` 需要将其 *CUDA* 接口转码为 *HIP* 接口；
