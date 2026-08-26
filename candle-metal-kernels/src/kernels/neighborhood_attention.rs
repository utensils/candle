use crate::utils::{BufferOffset, EncoderProvider};
use crate::{debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels};
use crate::{DType, MetalKernelError, Output, Source};
use objc2_metal::{MTLDevice, MTLSize};

#[allow(clippy::too_many_arguments)]
pub fn call_neighborhood_attention3d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch: usize,
    time: usize,
    height: usize,
    width: usize,
    heads: usize,
    head_dim: usize,
    kernel: [usize; 3],
    scale: f32,
    q: BufferOffset,
    k: BufferOffset,
    v: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        DType::F32 => "neighborhood_attention3d_f32",
        DType::F16 => "neighborhood_attention3d_f16",
        DType::BF16 => "neighborhood_attention3d_bf16",
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "non-floating",
                "neighborhood-attention-3d",
            ))
        }
    };
    let neighbors = kernel
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| {
            MetalKernelError::InvalidInput(
                "neighborhood-attention kernel volume overflows usize".to_string(),
            )
        })?;
    let threadgroup_bytes = neighbors
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            MetalKernelError::InvalidInput(
                "neighborhood-attention threadgroup byte count overflows usize".to_string(),
            )
        })?;
    let threadgroup_limit = device.as_ref().maxThreadgroupMemoryLength();
    if threadgroup_bytes > threadgroup_limit {
        return Err(MetalKernelError::InvalidInput(format!(
            "neighborhood-attention kernel {kernel:?} requires {threadgroup_bytes} bytes of threadgroup memory, but this Metal device supports {threadgroup_limit}"
        )));
    }

    let pipeline = kernels.load_pipeline(device, Source::NeighborhoodAttention, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "{name} {batch}x{time}x{height}x{width} h={heads} d={head_dim}"
    );
    set_params!(
        encoder,
        (
            &q,
            &k,
            &v,
            Output::new(output),
            time,
            height,
            width,
            heads,
            head_dim,
            kernel[0],
            kernel[1],
            kernel[2],
            scale
        )
    );
    encoder.set_threadgroup_memory_length(0, threadgroup_bytes);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: batch * time * height * width,
            height: heads,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
