#[cfg(feature = "metal")]
use crate::DType;
use crate::{CpuStorage, Layout, Result, Shape, Tensor};
use half::bf16;

const GROUP_SIZE: usize = 256;

#[derive(Debug, Clone, Copy)]
struct DequantizeInt8ConvRot256;

fn validate_layouts(packed: &Layout, scales: &Layout) -> Result<(usize, usize, usize)> {
    if !packed.is_contiguous() || !scales.is_contiguous() {
        crate::bail!("INT8 ConvRot inputs must be contiguous")
    }
    let [rows, cols] = packed.dims() else {
        crate::bail!("INT8 ConvRot packed input must have rank 2")
    };
    if *cols == 0 || cols % GROUP_SIZE != 0 {
        crate::bail!("INT8 ConvRot width {cols} must be divisible by {GROUP_SIZE}")
    }
    let scale_count = scales.shape().elem_count();
    if scale_count != 1 && scale_count != *rows {
        crate::bail!(
            "INT8 ConvRot requires one scale or one per row, got {scale_count} for {rows} rows"
        )
    }
    Ok((*rows, *cols, scale_count))
}

fn hadamard4(values: &mut [f32]) {
    let mut stride = 1;
    while stride < values.len() {
        let block = stride * 4;
        for base in (0..values.len()).step_by(block) {
            for offset in 0..stride {
                let i0 = base + offset;
                let i1 = i0 + stride;
                let i2 = i1 + stride;
                let i3 = i2 + stride;
                let (a, b, c, d) = (values[i0], values[i1], values[i2], values[i3]);
                values[i0] = a + b + c - d;
                values[i1] = a + b - c + d;
                values[i2] = a - b + c + d;
                values[i3] = -a + b + c + d;
            }
        }
        stride = block;
    }
}

impl crate::CustomOp2 for DequantizeInt8ConvRot256 {
    fn name(&self) -> &'static str {
        "dequantize-int8-convrot-256"
    }

    fn cpu_fwd(
        &self,
        packed: &CpuStorage,
        packed_layout: &Layout,
        scales: &CpuStorage,
        scales_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, cols, scale_count) = validate_layouts(packed_layout, scales_layout)?;
        let packed = packed.as_slice::<u8>()?;
        let packed = &packed[packed_layout.start_offset()..][..rows * cols];
        let scales = scales.as_slice::<f32>()?;
        let scales = &scales[scales_layout.start_offset()..][..scale_count];
        let mut output = Vec::with_capacity(rows * cols);
        let mut group = [0f32; GROUP_SIZE];
        for row in 0..rows {
            let scale = scales[if scale_count == 1 { 0 } else { row }] / 16.0;
            for packed_group in packed[row * cols..(row + 1) * cols].chunks_exact(GROUP_SIZE) {
                for (value, byte) in group.iter_mut().zip(packed_group) {
                    *value = (*byte as i8) as f32;
                }
                hadamard4(&mut group);
                output.extend(group.iter().map(|value| bf16::from_f32(value * scale)));
            }
        }
        Ok((CpuStorage::BF16(output), (rows, cols).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        packed: &crate::MetalStorage,
        packed_layout: &Layout,
        scales: &crate::MetalStorage,
        scales_layout: &Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        use crate::backend::BackendStorage;

        let (rows, cols, scale_count) = validate_layouts(packed_layout, scales_layout)?;
        if packed.dtype() != DType::U8 || scales.dtype() != DType::F32 {
            crate::bail!(
                "INT8 ConvRot expects U8 bytes and F32 scales, got {:?} and {:?}",
                packed.dtype(),
                scales.dtype()
            )
        }
        let device = packed.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(rows * cols, DType::BF16)
            .with_label("int8-convrot-bf16")
            .build()?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_dequantize_int8_convrot_256(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            cols,
            scale_count,
            crate::metal_backend::buffer_o(packed.buffer(), packed_layout, DType::U8),
            crate::metal_backend::buffer_o(scales.buffer(), scales_layout, DType::F32),
            &output,
        )
        .map_err(crate::Error::wrap)?;
        Ok((
            crate::MetalStorage::new(output, device.clone(), rows * cols, DType::BF16),
            (rows, cols).into(),
        ))
    }
}

/// Reconstruct tensorwise signed INT8 ConvRot weights in 256-value groups.
///
/// Metal execution allocates one BF16 output buffer and 1 KiB of threadgroup
/// memory per active group. It does not allocate full-size intermediate tensors.
pub fn dequantize_int8_convrot_256(packed: &Tensor, scales: &Tensor) -> Result<Tensor> {
    packed.apply_op2_no_bwd(scales, &DequantizeInt8ConvRot256)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    fn reference_input(rows: usize) -> Vec<u8> {
        (0..rows * GROUP_SIZE)
            .map(|index| ((index as i32 % 17) - 8) as i8 as u8)
            .collect()
    }

    #[test]
    fn cpu_reconstructs_one_group() -> Result<()> {
        let packed = Tensor::from_vec(reference_input(1), (1, GROUP_SIZE), &Device::Cpu)?;
        let scales = Tensor::new(&[0.25f32], &Device::Cpu)?;
        let output = dequantize_int8_convrot_256(&packed, &scales)?;
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.dims(), &[1, GROUP_SIZE]);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_cpu_for_tiny_bounded_inputs() -> Result<()> {
        let cpu_packed = Tensor::from_vec(reference_input(3), (3, GROUP_SIZE), &Device::Cpu)?;
        let cpu_scales = Tensor::new(&[0.25f32, 0.5, 0.75], &Device::Cpu)?;
        let expected = dequantize_int8_convrot_256(&cpu_packed, &cpu_scales)?;

        let metal = Device::new_metal(0)?;
        let actual = dequantize_int8_convrot_256(
            &cpu_packed.to_device(&metal)?,
            &cpu_scales.to_device(&metal)?,
        )?;
        metal.synchronize()?;
        assert_eq!(
            actual.to_device(&Device::Cpu)?.to_vec2::<bf16>()?,
            expected.to_vec2::<bf16>()?
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "explicit 192 MiB Metal residency smoke test"]
    fn metal_largest_ltx25_weight_stays_bounded() -> Result<()> {
        const ROWS: usize = 16_384;
        const COLS: usize = 4_096;
        let metal = Device::new_metal(0)?;
        let packed = Tensor::zeros((ROWS, COLS), DType::U8, &metal)?;
        let scales = Tensor::ones(ROWS, DType::F32, &metal)?;
        let output = dequantize_int8_convrot_256(&packed, &scales)?;
        metal.synchronize()?;
        assert_eq!(output.dims(), &[ROWS, COLS]);
        assert_eq!(output.dtype(), DType::BF16);
        Ok(())
    }
}
