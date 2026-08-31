#include "cuda_utils.cuh"
#include <stdint.h>

// Reconstruct tensorwise signed INT8 ConvRot weights in 256-value groups.
//
// This is a line-for-line port of `candle-metal-kernels/src/metal_src/convrot.metal`
// so the three backends stay bit-identical: one block handles one 256-wide
// group of one row, the butterfly runs three `stride *= 4` rounds over shared
// memory, and the only roundings are one power-of-two scale multiply and one
// round-to-nearest-even narrowing to BF16. Every butterfly intermediate is an
// exact integer below 2^24, so thread ordering cannot change the result.
//
// Grid: x = row, y = group (cols / 256); block: 256 threads. Rows sit on the
// x dimension because a quantized embedding table can exceed the 65,535-block
// limit of y.
extern "C" __global__ void dequantize_int8_convrot_256_bf16(
    const uint8_t *packed,
    const float *scales,
    __nv_bfloat16 *output,
    const size_t cols,
    const size_t scale_count) {
    __shared__ float values[256];
    const unsigned tid = threadIdx.x;
    const size_t row = blockIdx.x;
    const size_t column = (size_t)blockIdx.y * 256 + tid;
    const size_t index = row * cols + column;
    values[tid] = (float)((int8_t)packed[index]);
    __syncthreads();

    for (unsigned stride = 1; stride < 256; stride *= 4) {
        const unsigned block = stride * 4;
        const unsigned base = (tid / block) * block + (tid % stride);
        const unsigned lane = (tid / stride) % 4;
        const float a = values[base];
        const float b = values[base + stride];
        const float c = values[base + 2 * stride];
        const float d = values[base + 3 * stride];
        __syncthreads();
        float value;
        switch (lane) {
            case 0: value = a + b + c - d; break;
            case 1: value = a + b - c + d; break;
            case 2: value = a - b + c + d; break;
            default: value = -a + b + c + d; break;
        }
        values[tid] = value;
        __syncthreads();
    }

    const float scale = scales[scale_count == 1 ? 0 : row] * 0.0625f;
    output[index] = __float2bfloat16_rn(values[tid] * scale);
}
