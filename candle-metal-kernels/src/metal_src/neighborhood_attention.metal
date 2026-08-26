#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void neighborhood_attention3d(
    device const T *q,
    device const T *k,
    device const T *v,
    device T *output,
    constant size_t &time,
    constant size_t &height,
    constant size_t &width,
    constant size_t &heads,
    constant size_t &head_dim,
    constant size_t &kernel_t,
    constant size_t &kernel_h,
    constant size_t &kernel_w,
    constant float &scale,
    threadgroup float *scores [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const uint threads = threads_per_group.x;
    const size_t query_linear = group.x;
    const size_t head = group.y;
    const size_t spatial = time * height * width;
    const size_t batch = query_linear / spatial;
    const size_t position = query_linear % spatial;
    const size_t query_t = position / (height * width);
    const size_t query_h = (position / width) % height;
    const size_t query_w = position % width;

    const size_t start_t = min(max(long(query_t) - long(kernel_t / 2), 0l), long(time - kernel_t));
    const size_t start_h = min(max(long(query_h) - long(kernel_h / 2), 0l), long(height - kernel_h));
    const size_t start_w = min(max(long(query_w) - long(kernel_w / 2), 0l), long(width - kernel_w));
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
            dot += float(q[query_base + d]) * float(k[key_base + d]);
        }
        scores[neighbor] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float maximum = -INFINITY;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            maximum = max(maximum, scores[neighbor]);
        }
        float denominator = 0.0f;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            const float weight = exp(scores[neighbor] - maximum);
            scores[neighbor] = weight;
            denominator += weight;
        }
        const float reciprocal = 1.0f / denominator;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            scores[neighbor] *= reciprocal;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t d = tid; d < head_dim; d += threads) {
        float value = 0.0f;
        for (size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            const size_t dt = neighbor / (kernel_h * kernel_w);
            const size_t dh = (neighbor / kernel_w) % kernel_h;
            const size_t dw = neighbor % kernel_w;
            const size_t value_position = ((start_t + dt) * height + start_h + dh) * width + start_w + dw;
            const size_t value_base = ((batch * spatial + value_position) * heads + head) * head_dim;
            value += scores[neighbor] * float(v[value_base + d]);
        }
        output[query_base + d] = T(value);
    }
}

#define instantiate_na3d(name, type) \
    template [[host_name(name)]] [[kernel]] decltype(neighborhood_attention3d<type>) neighborhood_attention3d<type>;

instantiate_na3d("neighborhood_attention3d_f32", float)
instantiate_na3d("neighborhood_attention3d_f16", half)
#if __METAL_VERSION__ >= 310
instantiate_na3d("neighborhood_attention3d_bf16", bfloat)
#endif
