#include "cuda_utils.cuh"
#include <stdint.h>

// CUDA twin of `candle-metal-kernels/src/metal_src/neighborhood_attention.metal`.
// One block computes every attention output for one (batch, query-position,
// head) triple: grid.x walks the flattened (batch, time, height, width)
// query positions, grid.y walks heads, and the fixed 256-thread block runs
// the same three phases as the Metal kernel behind the same two
// `__syncthreads()` barriers — score every neighbor into dynamic shared
// memory, softmax it on thread 0 (the neighbor count is always small, e.g.
// the largest checkpoint kernel is 3x7x7 = 147, so a single-thread softmax
// costs nothing next to the dot products), then have every thread
// accumulate a strided slice of the head dimension. Every arithmetic step
// mirrors the Metal source line-for-line so the two backends stay within
// the parity test's tolerance.
template <typename T>
__device__ void neighborhood_attention3d(
    const T *q,
    const T *k,
    const T *v,
    T *output,
    const size_t time,
    const size_t height,
    const size_t width,
    const size_t heads,
    const size_t head_dim,
    const size_t kernel_t,
    const size_t kernel_h,
    const size_t kernel_w,
    const float scale) {
    extern __shared__ float scores[];
    const unsigned tid = threadIdx.x;
    const unsigned threads = blockDim.x;
    const size_t query_linear = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t spatial = time * height * width;
    const size_t batch = query_linear / spatial;
    const size_t position = query_linear % spatial;
    const size_t query_t = position / (height * width);
    const size_t query_h = (position / width) % height;
    const size_t query_w = position % width;

    const long start_t_signed =
        min(max((long)query_t - (long)(kernel_t / 2), 0l), (long)(time - kernel_t));
    const long start_h_signed =
        min(max((long)query_h - (long)(kernel_h / 2), 0l), (long)(height - kernel_h));
    const long start_w_signed =
        min(max((long)query_w - (long)(kernel_w / 2), 0l), (long)(width - kernel_w));
    const size_t start_t = (size_t)start_t_signed;
    const size_t start_h = (size_t)start_h_signed;
    const size_t start_w = (size_t)start_w_signed;
    const size_t neighbors = kernel_t * kernel_h * kernel_w;
    const size_t query_base = (query_linear * heads + head) * head_dim;

    for (size_t neighbor = tid; neighbor < neighbors; neighbor += threads) {
        const size_t dt = neighbor / (kernel_h * kernel_w);
        const size_t dh = (neighbor / kernel_w) % kernel_h;
        const size_t dw = neighbor % kernel_w;
        const size_t key_position = ((start_t + dt) * height + start_h + dh) * width + start_w + dw;
        const size_t key_base = ((batch * spatial + key_position) * heads + head) * head_dim;
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            dot += static_cast<float>(q[query_base + d]) * static_cast<float>(k[key_base + d]);
        }
        scores[neighbor] = dot * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float maximum = -INFINITY;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            maximum = max(maximum, scores[neighbor]);
        }
        float denominator = 0.0f;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            const float weight = expf(scores[neighbor] - maximum);
            scores[neighbor] = weight;
            denominator += weight;
        }
        const float reciprocal = 1.0f / denominator;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            scores[neighbor] *= reciprocal;
        }
    }
    __syncthreads();

    for (size_t d = tid; d < head_dim; d += threads) {
        float value = 0.0f;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            const size_t dt = neighbor / (kernel_h * kernel_w);
            const size_t dh = (neighbor / kernel_w) % kernel_h;
            const size_t dw = neighbor % kernel_w;
            const size_t value_position =
                ((start_t + dt) * height + start_h + dh) * width + start_w + dw;
            const size_t value_base = ((batch * spatial + value_position) * heads + head) * head_dim;
            value += scores[neighbor] * static_cast<float>(v[value_base + d]);
        }
        output[query_base + d] = static_cast<T>(value);
    }
}

#define NEIGHBORHOOD_ATTENTION3D_OP(TYPENAME, FN_NAME)                                  \
    extern "C" __global__ void FN_NAME(                                                 \
        const TYPENAME *q,                                                              \
        const TYPENAME *k,                                                              \
        const TYPENAME *v,                                                              \
        TYPENAME *output,                                                               \
        const size_t time,                                                              \
        const size_t height,                                                            \
        const size_t width,                                                             \
        const size_t heads,                                                             \
        const size_t head_dim,                                                          \
        const size_t kernel_t,                                                          \
        const size_t kernel_h,                                                          \
        const size_t kernel_w,                                                          \
        const float scale) {                                                            \
        neighborhood_attention3d<TYPENAME>(                                             \
            q, k, v, output, time, height, width, heads, head_dim, kernel_t, kernel_h,   \
            kernel_w, scale);                                                           \
    }

NEIGHBORHOOD_ATTENTION3D_OP(float, neighborhood_attention3d_f32)
NEIGHBORHOOD_ATTENTION3D_OP(__half, neighborhood_attention3d_f16)

#if __CUDA_ARCH__ >= 800
NEIGHBORHOOD_ATTENTION3D_OP(__nv_bfloat16, neighborhood_attention3d_bf16)
#endif
