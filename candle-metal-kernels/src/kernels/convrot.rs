use crate::utils::{BufferOffset, EncoderProvider};
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

/// Reconstructs tensorwise INT8 ConvRot rows into a single BF16 output buffer.
///
/// The kernel uses exactly 1 KiB of threadgroup memory and allocates no temporary
/// device buffers. Each 256-thread group owns one 256-column section of one row.
pub fn call_dequantize_int8_convrot_256(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    cols: usize,
    scale_count: usize,
    packed: BufferOffset,
    scales: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline =
        kernels.load_pipeline(device, Source::ConvRot, "dequantize_int8_convrot_256_bf16")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "dequantize_int8_convrot_256_bf16 rows={rows} cols={cols}"
    );
    set_params!(
        encoder,
        (&packed, &scales, Output::new(output), cols, scale_count)
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: cols / 256,
            height: rows,
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
