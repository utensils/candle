use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};

#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = shape[0] * l_out * shape[1] * k_size;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col1d {name} dst_el={dst_el}");
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            k_size,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_col2im1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    k_size: usize,
    stride: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_in = shape[1];
    let c_out = shape[2];
    let l_out = (l_in - 1) * stride + k_size;
    let dst_el = shape[0] * c_out * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "col2im1d {name} dst_el={dst_el}");
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            l_in,
            c_out,
            k_size,
            stride,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn im2col_geometry(
    shape: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
) -> Result<(usize, usize, usize), MetalKernelError> {
    let error =
        || MetalKernelError::InvalidInput("im2col invalid or overflowing dimensions".into());
    if shape.len() != 4 || stride == 0 || dilation == 0 || h_k == 0 || w_k == 0 {
        return Err(error());
    }
    let axis = |input: usize, kernel: usize| -> Option<usize> {
        input
            .checked_add(padding.checked_mul(2)?)?
            .checked_sub(dilation.checked_mul(kernel.checked_sub(1)?)?)?
            .checked_sub(1)?
            .checked_div(stride)?
            .checked_add(1)
    };
    let h = axis(shape[2], h_k).ok_or_else(error)?;
    let w = axis(shape[3], w_k).ok_or_else(error)?;
    let rows = shape[0]
        .checked_mul(h)
        .and_then(|v| v.checked_mul(w))
        .ok_or_else(error)?;
    Ok((rows, h, w))
}
fn im2col_range_geometry(
    shape: &[usize],
    params: (usize, usize, usize, usize, usize),
    offset: usize,
    count: usize,
) -> Result<(usize, usize, usize), MetalKernelError> {
    let error =
        || MetalKernelError::InvalidInput("im2col invalid spatial range or uint dispatch".into());
    let (rows, h, w) = im2col_geometry(shape, params)?;
    if count == 0 || offset.checked_add(count).is_none_or(|end| end > rows) {
        return Err(error());
    }
    let patch = shape[1]
        .checked_mul(params.0)
        .and_then(|v| v.checked_mul(params.1))
        .ok_or_else(error)?;
    rows.checked_mul(patch).ok_or_else(error)?;
    let elements = count.checked_mul(patch).ok_or_else(error)?;
    if elements == 0 || elements > u32::MAX as usize - 1023 {
        return Err(error());
    }
    Ok((elements, h, w))
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let (rows, _, _) = im2col_geometry(shape, (h_k, w_k, stride, padding, dilation))?;
    call_im2col_strided_range(
        device,
        ep,
        kernels,
        name,
        shape,
        strides,
        (h_k, w_k, stride, padding, dilation),
        0,
        rows,
        input,
        output,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided_range(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    spatial_offset: usize,
    spatial_count: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let (dst_el, h_out, w_out) = im2col_range_geometry(
        shape,
        (h_k, w_k, stride, padding, dilation),
        spatial_offset,
        spatial_count,
    )?;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col {name} dst_el={dst_el}");
    set_params!(
        encoder,
        (
            dst_el,
            spatial_offset,
            h_out,
            w_out,
            h_k,
            w_k,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

fn scatter_geometry(
    rows: usize,
    offset: usize,
    spatial: usize,
    channels: usize,
) -> Result<usize, MetalKernelError> {
    let error =
        || MetalKernelError::InvalidInput("conv scatter invalid or overflowing dimensions".into());
    if rows == 0 || spatial == 0 || channels == 0 {
        return Err(error());
    }
    let count = rows.checked_mul(channels).ok_or_else(error)?;
    let last = offset.checked_add(rows - 1).ok_or_else(error)?;
    // Largest NCHW index touched, including a partial final batch.
    (last / spatial)
        .checked_mul(channels)
        .and_then(|v| v.checked_add(channels - 1))
        .and_then(|v| v.checked_mul(spatial))
        .and_then(|v| v.checked_add(last % spatial))
        .ok_or_else(error)?;
    if count > u32::MAX as usize - 1023 {
        return Err(error());
    }
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv2d_scatter(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    rows: usize,
    offset: usize,
    spatial: usize,
    channels: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let count = scatter_geometry(rows, offset, spatial, channels)?;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (groups, threads) = linear_split(&pipeline, count);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (count, offset, spatial, channels, input, Output::new(output))
    );
    encoder.dispatch_thread_groups(groups, threads);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_nearest_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let scale_w = shape[2] as f32 / out_w as f32;
    let scale_h = shape[3] as f32 / out_h as f32;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "upsample_nearest2d {name} {out_w}x{out_h}");
    set_params!(
        encoder,
        (
            out_w,
            out_h,
            scale_w,
            scale_h,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_bilinear_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let dst_el = out_w * out_h * shape[0] * shape[1];

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "upsample_bilinear2d {name} {out_w}x{out_h}");

    set_params!(
        encoder,
        (
            out_w,
            out_h,
            align_corners,
            scale_h.is_some(),
            scale_h.unwrap_or(0.0) as f32,
            scale_w.is_some(),
            scale_w.unwrap_or(0.0) as f32,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_pool2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "pool2d {name} {out_w}x{out_h} k={w_k}x{h_k}");
    set_params!(
        encoder,
        (
            w_k,
            h_k,
            w_stride,
            h_stride,
            shape,
            strides,
            input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    c_out: usize,
    l_out: usize,
    b_size: usize,
    src_shape: &[usize],
    src_strides: &[usize],
    kernel_shape: &[usize],
    kernel_strides: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = c_out * l_out * b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "conv_transpose1d {name} c_out={c_out} l_out={l_out} b={b_size}"
    );
    set_params!(
        encoder,
        (
            l_out,
            stride,
            padding,
            out_padding,
            dilation,
            src_shape,
            src_strides,
            kernel_shape,
            kernel_strides,
            (input, input_offset),
            (kernel, kernel_offset),
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub struct CallConvTranspose2dCfg<'a> {
    pub dilation: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub c_out: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub b_size: usize,
    pub input_dims: &'a [usize],
    pub input_stride: &'a [usize],
    pub kernel_dims: &'a [usize],
    pub kernel_stride: &'a [usize],
    pub input_offset: usize,
    pub kernel_offset: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: CallConvTranspose2dCfg,
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = cfg.c_out * cfg.out_w * cfg.out_h * cfg.b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "conv_transpose2d {name} c_out={} {}x{} b={}",
        cfg.c_out,
        cfg.out_w,
        cfg.out_h,
        cfg.b_size
    );
    set_params!(
        encoder,
        (
            cfg.out_w,
            cfg.out_h,
            cfg.stride,
            cfg.padding,
            cfg.output_padding,
            cfg.dilation,
            cfg.input_dims,
            cfg.input_stride,
            cfg.kernel_dims,
            cfg.kernel_stride,
            (input, cfg.input_offset),
            (kernel, cfg.kernel_offset),
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[cfg(test)]
mod range_tests {
    use super::*;
    #[test]
    fn scatter_rejects_invalid_ranges() {
        for (r, o, s, c) in [
            (0, 0, 1, 1),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
            (2, usize::MAX, 1, 1),
            (1, usize::MAX / 2, 1, 3),
            (usize::MAX, 0, 1, 2),
        ] {
            assert!(scatter_geometry(r, o, s, c).is_err());
        }
        assert_eq!(
            scatter_geometry(13, (1usize << 32) + 7, 65536, 3).unwrap(),
            39
        );
    }
    #[test]
    fn range_uses_size_t_offset_beyond_uint() {
        let result = im2col_range_geometry(
            &[2, 1, 65536, 65536],
            (1, 1, 1, 0, 1),
            (1usize << 32) + 7,
            13,
        )
        .unwrap();
        assert_eq!(result, (13, 65536, 65536));
    }
    #[test]
    fn rejects_empty_out_of_bounds_and_overflow() {
        for (offset, count) in [(0, 0), (99, 2), (usize::MAX, 1)] {
            assert!(
                im2col_range_geometry(&[1, 3, 10, 10], (1, 1, 1, 0, 1), offset, count).is_err()
            );
        }
        assert!(im2col_geometry(&[1, 3, usize::MAX, 10], (3, 3, 1, 1, 1)).is_err());
        assert!(im2col_geometry(&[1, 3, 10, 10], (3, 3, 0, 1, 1)).is_err());
        assert!(
            im2col_range_geometry(&[1, 2304, 1368, 1368], (1, 1, 1, 0, 1), 0, 1368 * 1368).is_err()
        );
    }
}
