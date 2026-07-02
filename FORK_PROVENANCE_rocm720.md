# ROCm7.2 instrumented mori fork (branch rocm720-mori-bf99bdf)
Base: ROCm/mori @ bf99bdf18fc69887a346913ca01c315c2aa9bd4c (>= 42e8954 / PR #366
"mlx5 collapsed CQ + dedicated dispatch send buffer for internode-v1", contained in
lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260702).
Overlay (ADDITIVE, env-gated MORI_ROCTX / MORI_ROCTX_TRANSFER; default OFF):
  src/io/roctx_mori.hpp            (new)  self-contained roctx range/mark helpers + bytes token
  src/io/engine.cpp                (+5)   host-send ranges at IOEngine[Session]::BatchWrite
  src/io/rdma/common.cpp           (+27)  batch_post range + bytes accumulator + async post->cq start
  src/io/rdma/backend_impl.cpp     (+7)   async post->cq range stop at ProcessOneCqe CQE sites
Baked into image rocmshared/pytorch-private:sglang_mori_roctx_cq_bytes_rocm720_20260702.
