#include <metal_stdlib>

using namespace metal;

#if __METAL_VERSION__ >= 310
kernel void dequantize_int8_convrot_256_bf16(
    device const uchar *packed,
    device const float *scales,
    device bfloat *output,
    constant size_t &cols,
    constant size_t &scale_count,
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    const size_t row = group.y;
    const size_t column = group.x * 256 + tid;
    const size_t index = row * cols + column;
    values[tid] = float(as_type<char>(packed[index]));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < 256; stride *= 4) {
        const uint block = stride * 4;
        const uint base = (tid / block) * block + (tid % stride);
        const uint lane = (tid / stride) % 4;
        const float a = values[base];
        const float b = values[base + stride];
        const float c = values[base + 2 * stride];
        const float d = values[base + 3 * stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        switch (lane) {
            case 0: values[tid] = a + b + c - d; break;
            case 1: values[tid] = a + b - c + d; break;
            case 2: values[tid] = a - b + c + d; break;
            default: values[tid] = -a + b + c + d; break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = scales[scale_count == 1 ? 0 : row] * 0.0625f;
    output[index] = bfloat(values[tid] * scale);
}
#endif
