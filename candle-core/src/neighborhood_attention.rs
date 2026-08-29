#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::DType;
use crate::{CpuStorage, Layout, Result, Shape, Tensor};

#[derive(Debug, Clone, Copy)]
struct NeighborhoodAttention3d {
    kernel: [usize; 3],
    scale: f32,
}

fn validate(q: &Layout, k: &Layout, v: &Layout, kernel: [usize; 3]) -> Result<[usize; 6]> {
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        crate::bail!("neighborhood attention requires contiguous q, k, and v")
    }
    if q.dims() != k.dims() || q.dims() != v.dims() {
        crate::bail!("neighborhood attention q, k, and v shapes must match")
    }
    let [batch, time, height, width, heads, head_dim] = q.dims() else {
        crate::bail!("neighborhood attention expects [B,T,H,W,heads,head_dim]")
    };
    if kernel.iter().any(|size| *size == 0 || size % 2 == 0) {
        crate::bail!("neighborhood attention kernels must be positive odd sizes")
    }
    if *head_dim == 0 || *head_dim > 256 {
        crate::bail!("neighborhood attention head dimension must be in 1..=256")
    }
    Ok([*batch, *time, *height, *width, *heads, *head_dim])
}

/// Clamp each kernel axis to its grid extent.
///
/// A requested kernel wider than the axis it slides over cannot describe a
/// real sliding window on that axis: NATTEN boundary-window semantics
/// degrade to full attention there instead (every query on that axis
/// attends across the whole axis, `start = 0`). The requested `kernel`
/// itself must stay a positive odd triple (checked by `validate`); the
/// clamped value returned here is what every backend actually loops over,
/// and it may be even when an axis extent is even.
fn effective_kernel(kernel: [usize; 3], time: usize, height: usize, width: usize) -> [usize; 3] {
    [kernel[0].min(time), kernel[1].min(height), kernel[2].min(width)]
}

impl crate::CustomOp3 for NeighborhoodAttention3d {
    fn name(&self) -> &'static str {
        "neighborhood-attention-3d"
    }

    fn cpu_fwd(
        &self,
        q: &CpuStorage,
        q_layout: &Layout,
        k: &CpuStorage,
        k_layout: &Layout,
        v: &CpuStorage,
        v_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let [batch, time, height, width, heads, head_dim] =
            validate(q_layout, k_layout, v_layout, self.kernel)?;
        if !matches!(
            (q, k, v),
            (CpuStorage::F32(_), CpuStorage::F32(_), CpuStorage::F32(_))
        ) {
            crate::bail!(
                "CPU neighborhood attention supports matching F32 q, k, and v tensors only"
            )
        }
        let q = q.as_slice::<f32>()?;
        let k = k.as_slice::<f32>()?;
        let v = v.as_slice::<f32>()?;
        let q = &q[q_layout.start_offset()..];
        let k = &k[k_layout.start_offset()..];
        let v = &v[v_layout.start_offset()..];
        let kernel = effective_kernel(self.kernel, time, height, width);
        let spatial = time * height * width;
        let neighbors: usize = kernel.iter().product();
        let mut output = vec![0f32; batch * spatial * heads * head_dim];
        let mut scores = vec![0f32; neighbors];
        for b in 0..batch {
            for qt in 0..time {
                let start_t = qt.saturating_sub(kernel[0] / 2).min(time - kernel[0]);
                for qh in 0..height {
                    let start_h = qh.saturating_sub(kernel[1] / 2).min(height - kernel[1]);
                    for qw in 0..width {
                        let start_w = qw.saturating_sub(kernel[2] / 2).min(width - kernel[2]);
                        let query_position = (qt * height + qh) * width + qw;
                        for head in 0..heads {
                            let query_base =
                                ((b * spatial + query_position) * heads + head) * head_dim;
                            let mut index = 0;
                            for dt in 0..kernel[0] {
                                for dh in 0..kernel[1] {
                                    for dw in 0..kernel[2] {
                                        let key_position = ((start_t + dt) * height + start_h + dh)
                                            * width
                                            + start_w
                                            + dw;
                                        let key_base = ((b * spatial + key_position) * heads
                                            + head)
                                            * head_dim;
                                        scores[index] = q[query_base..query_base + head_dim]
                                            .iter()
                                            .zip(&k[key_base..key_base + head_dim])
                                            .map(|(q, k)| q * k)
                                            .sum::<f32>()
                                            * self.scale;
                                        index += 1;
                                    }
                                }
                            }
                            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let denominator: f32 = scores
                                .iter_mut()
                                .map(|score| {
                                    *score = (*score - maximum).exp();
                                    *score
                                })
                                .sum();
                            for score in &mut scores {
                                *score /= denominator;
                            }
                            for d in 0..head_dim {
                                let mut value = 0f32;
                                let mut index = 0;
                                for dt in 0..kernel[0] {
                                    for dh in 0..kernel[1] {
                                        for dw in 0..kernel[2] {
                                            let value_position =
                                                ((start_t + dt) * height + start_h + dh) * width
                                                    + start_w
                                                    + dw;
                                            let value_base =
                                                ((b * spatial + value_position) * heads + head)
                                                    * head_dim;
                                            value += scores[index] * v[value_base + d];
                                            index += 1;
                                        }
                                    }
                                }
                                output[query_base + d] = value;
                            }
                        }
                    }
                }
            }
        }
        Ok((CpuStorage::F32(output), q_layout.shape().clone()))
    }

    /// CUDA twin of `metal_fwd`: one 256-thread block per (query position,
    /// head), the same shared-memory score/softmax/accumulate phases as
    /// `neighborhood_attention.metal`, one output buffer per dtype, no host
    /// copy. Dispatches through `Map3` so the same code path handles F32,
    /// F16, and BF16 without a per-dtype `unsafe` block.
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        q: &crate::CudaStorage,
        q_layout: &Layout,
        k: &crate::CudaStorage,
        k_layout: &Layout,
        v: &crate::CudaStorage,
        v_layout: &Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
        };
        use crate::cuda_backend::{kernel_name, kernels, Map3, WrapErr};
        use crate::{CudaDevice, WithDType};

        let [batch, time, height, width, heads, head_dim] =
            validate(q_layout, k_layout, v_layout, self.kernel)?;
        if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
            crate::bail!("neighborhood attention q, k, and v dtypes must match")
        }
        if !matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            crate::bail!("unsupported neighborhood attention dtype {:?}", q.dtype())
        }

        let kernel = effective_kernel(self.kernel, time, height, width);
        let neighbors: usize = kernel.iter().product();
        let threadgroup_bytes = neighbors * std::mem::size_of::<f32>();
        // Mirror the Metal backend's pre-dispatch guard: refuse before
        // touching the device rather than surfacing an opaque CUDA launch
        // failure. 48 KiB is the per-block dynamic shared memory budget
        // guaranteed on every supported compute capability without an
        // explicit `cudaFuncSetAttribute` opt-in.
        const MAX_DEFAULT_SHARED_MEM_BYTES: usize = 48 * 1024;
        if threadgroup_bytes > MAX_DEFAULT_SHARED_MEM_BYTES {
            crate::bail!(
                "neighborhood attention kernel {:?} requires {threadgroup_bytes} bytes of shared memory, but the default CUDA per-block budget is {MAX_DEFAULT_SHARED_MEM_BYTES}",
                kernel
            )
        }

        struct Launch {
            batch: usize,
            time: usize,
            height: usize,
            width: usize,
            heads: usize,
            head_dim: usize,
            kernel: [usize; 3],
            scale: f32,
            threadgroup_bytes: usize,
        }

        impl Map3 for Launch {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                &self,
                q: &CudaSlice<T>,
                q_layout: &Layout,
                k: &CudaSlice<T>,
                k_layout: &Layout,
                v: &CudaSlice<T>,
                v_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let q = match q_layout.contiguous_offsets() {
                    Some((o1, o2)) => q.slice(o1..o2),
                    None => crate::bail!("neighborhood attention requires contiguous q, k, and v"),
                };
                let k = match k_layout.contiguous_offsets() {
                    Some((o1, o2)) => k.slice(o1..o2),
                    None => crate::bail!("neighborhood attention requires contiguous q, k, and v"),
                };
                let v = match v_layout.contiguous_offsets() {
                    Some((o1, o2)) => v.slice(o1..o2),
                    None => crate::bail!("neighborhood attention requires contiguous q, k, and v"),
                };
                let elements =
                    self.batch * self.time * self.height * self.width * self.heads * self.head_dim;
                let func = dev.get_or_load_func(
                    &kernel_name::<T>("neighborhood_attention3d"),
                    &kernels::NEIGHBORHOOD_ATTENTION,
                )?;
                // SAFETY: every element is written by exactly one thread
                // before the buffer is observed; the kernel covers
                // `elements` outputs exactly.
                let output = unsafe { dev.alloc::<T>(elements)? };
                let cfg = LaunchConfig {
                    grid_dim: (
                        (self.batch * self.time * self.height * self.width) as u32,
                        self.heads as u32,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: self.threadgroup_bytes as u32,
                };
                let time = self.time as u64;
                let height = self.height as u64;
                let width = self.width as u64;
                let heads = self.heads as u64;
                let head_dim = self.head_dim as u64;
                let kernel_t = self.kernel[0] as u64;
                let kernel_h = self.kernel[1] as u64;
                let kernel_w = self.kernel[2] as u64;
                let scale = self.scale;
                let stream = dev.cuda_stream();
                let mut builder = stream.launch_builder(&func);
                builder
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&output)
                    .arg(&time)
                    .arg(&height)
                    .arg(&width)
                    .arg(&heads)
                    .arg(&head_dim)
                    .arg(&kernel_t)
                    .arg(&kernel_h)
                    .arg(&kernel_w)
                    .arg(&scale);
                // SAFETY: argument types and launch geometry match the
                // kernel signature.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(output)
            }
        }

        let device = q.device().clone();
        let slice = Launch {
            batch,
            time,
            height,
            width,
            heads,
            head_dim,
            kernel,
            scale: self.scale,
            threadgroup_bytes,
        }
        .map(&q.slice, q_layout, &k.slice, k_layout, &v.slice, v_layout, &device)?;
        Ok((
            crate::cuda_backend::CudaStorage { slice, device },
            q_layout.shape().clone(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &crate::MetalStorage,
        q_layout: &Layout,
        k: &crate::MetalStorage,
        k_layout: &Layout,
        v: &crate::MetalStorage,
        v_layout: &Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        use crate::backend::BackendStorage;
        let [batch, time, height, width, heads, head_dim] =
            validate(q_layout, k_layout, v_layout, self.kernel)?;
        if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
            crate::bail!("neighborhood attention q, k, and v dtypes must match")
        }
        if !matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            crate::bail!("unsupported neighborhood attention dtype {:?}", q.dtype())
        }
        let kernel = effective_kernel(self.kernel, time, height, width);
        let device = q.device();
        let elements = q_layout.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, q.dtype())
            .with_label("neighborhood-attention-3d")
            .build()?;
        let encoder = device.command_encoder()?;
        let metal_dtype = match q.dtype() {
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            _ => unreachable!("dtype validated above"),
        };
        candle_metal_kernels::call_neighborhood_attention3d(
            device.metal_device(),
            &encoder,
            device.kernels(),
            metal_dtype,
            batch,
            time,
            height,
            width,
            heads,
            head_dim,
            kernel,
            self.scale,
            crate::metal_backend::buffer_o(q.buffer(), q_layout, q.dtype()),
            crate::metal_backend::buffer_o(k.buffer(), k_layout, k.dtype()),
            crate::metal_backend::buffer_o(v.buffer(), v_layout, v.dtype()),
            &output,
        )
        .map_err(crate::Error::wrap)?;
        Ok((
            crate::MetalStorage::new(output, device.clone(), elements, q.dtype()),
            q_layout.shape().clone(),
        ))
    }
}

/// Fused 3D neighborhood attention with NATTEN boundary-window semantics.
pub fn neighborhood_attention3d(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kernel: [usize; 3],
    scale: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(k, v, &NeighborhoodAttention3d { kernel, scale })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    #[test]
    fn one_voxel_window_returns_values() -> Result<()> {
        let q = Tensor::ones((1, 1, 1, 3, 1, 2), DType::F32, &Device::Cpu)?;
        let k = q.clone();
        let v = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6.],
            (1, 1, 1, 3, 1, 2),
            &Device::Cpu,
        )?;
        let output = neighborhood_attention3d(&q, &k, &v, [1, 1, 1], 1.0)?;
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            v.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn shifted_boundary_window_keeps_constant_values() -> Result<()> {
        let q = Tensor::ones((1, 3, 3, 3, 1, 2), DType::F32, &Device::Cpu)?;
        let k = q.clone();
        let v = Tensor::full(7f32, (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let output = neighborhood_attention3d(&q, &k, &v, [3, 3, 3], 0.5)?;
        assert!(output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| (*value - 7.0).abs() < 1e-5));
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_cpu_for_shifted_boundaries() -> Result<()> {
        let values: Vec<f32> = (0..54).map(|index| index as f32 / 10.0).collect();
        let q = Tensor::from_vec(values.clone(), (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let k = Tensor::from_vec(
            values.iter().rev().copied().collect::<Vec<_>>(),
            (1, 3, 3, 3, 1, 2),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(values, (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let expected = neighborhood_attention3d(&q, &k, &v, [3, 3, 3], 2f32.sqrt().recip())?;

        let metal = Device::new_metal(0)?;
        let actual = neighborhood_attention3d(
            &q.to_device(&metal)?,
            &k.to_device(&metal)?,
            &v.to_device(&metal)?,
            [3, 3, 3],
            2f32.sqrt().recip(),
        )?;
        metal.synchronize()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-4));
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_half_types_match_f32_reference() -> Result<()> {
        let values: Vec<f32> = (0..54).map(|index| index as f32 / 20.0).collect();
        let q = Tensor::from_vec(values.clone(), (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let k = Tensor::from_vec(
            values.iter().rev().copied().collect::<Vec<_>>(),
            (1, 3, 3, 3, 1, 2),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(values, (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let expected = neighborhood_attention3d(&q, &k, &v, [3, 3, 3], 2f32.sqrt().recip())?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let metal = Device::new_metal(0)?;
        for (dtype, tolerance) in [(DType::F16, 3e-3), (DType::BF16, 3e-2)] {
            let actual = neighborhood_attention3d(
                &q.to_dtype(dtype)?.to_device(&metal)?,
                &k.to_dtype(dtype)?.to_device(&metal)?,
                &v.to_dtype(dtype)?.to_device(&metal)?,
                [3, 3, 3],
                2f32.sqrt().recip(),
            )?;
            metal.synchronize()?;
            let actual = actual
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(expected, actual)| (expected - actual).abs() < tolerance),
                "{dtype:?} Metal neighborhood attention exceeded tolerance {tolerance}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_rejects_kernel_above_threadgroup_memory_limit() -> Result<()> {
        let metal = Device::new_metal(0)?;
        let tensor = Tensor::ones((1, 33, 33, 33, 1, 1), DType::F32, &metal)?;
        let error = neighborhood_attention3d(&tensor, &tensor, &tensor, [33, 33, 33], 1.0)
            .expect_err("oversized threadgroup allocation must fail before dispatch");
        assert!(
            error.to_string().contains("threadgroup memory"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_cpu_for_shifted_boundaries() -> Result<()> {
        let values: Vec<f32> = (0..54).map(|index| index as f32 / 10.0).collect();
        let q = Tensor::from_vec(values.clone(), (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let k = Tensor::from_vec(
            values.iter().rev().copied().collect::<Vec<_>>(),
            (1, 3, 3, 3, 1, 2),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(values, (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let expected = neighborhood_attention3d(&q, &k, &v, [3, 3, 3], 2f32.sqrt().recip())?;

        let cuda = Device::new_cuda(0)?;
        let actual = neighborhood_attention3d(
            &q.to_device(&cuda)?,
            &k.to_device(&cuda)?,
            &v.to_device(&cuda)?,
            [3, 3, 3],
            2f32.sqrt().recip(),
        )?;
        cuda.synchronize()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-4));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_cpu_for_an_asymmetric_kernel() -> Result<()> {
        // Mirrors the diffusion VAE's real kernel shapes (e.g. [3, 7, 7]),
        // where every axis clamps its window start independently and the
        // grid dimensions differ from one another.
        let elements = 5 * 7 * 9 * 2;
        let values: Vec<f32> = (0..elements)
            .map(|index| (index as f32 % 23.0) / 4.0)
            .collect();
        let q = Tensor::from_vec(values.clone(), (1, 5, 7, 9, 1, 2), &Device::Cpu)?;
        let k = Tensor::from_vec(
            values.iter().rev().copied().collect::<Vec<_>>(),
            (1, 5, 7, 9, 1, 2),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(values, (1, 5, 7, 9, 1, 2), &Device::Cpu)?;
        let expected = neighborhood_attention3d(&q, &k, &v, [3, 5, 7], 2f32.sqrt().recip())?;

        let cuda = Device::new_cuda(0)?;
        let actual = neighborhood_attention3d(
            &q.to_device(&cuda)?,
            &k.to_device(&cuda)?,
            &v.to_device(&cuda)?,
            [3, 5, 7],
            2f32.sqrt().recip(),
        )?;
        cuda.synchronize()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-3));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_half_types_match_f32_reference() -> Result<()> {
        let values: Vec<f32> = (0..54).map(|index| index as f32 / 20.0).collect();
        let q = Tensor::from_vec(values.clone(), (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let k = Tensor::from_vec(
            values.iter().rev().copied().collect::<Vec<_>>(),
            (1, 3, 3, 3, 1, 2),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(values, (1, 3, 3, 3, 1, 2), &Device::Cpu)?;
        let expected = neighborhood_attention3d(&q, &k, &v, [3, 3, 3], 2f32.sqrt().recip())?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let cuda = Device::new_cuda(0)?;
        for (dtype, tolerance) in [(DType::F16, 3e-3), (DType::BF16, 3e-2)] {
            let actual = neighborhood_attention3d(
                &q.to_dtype(dtype)?.to_device(&cuda)?,
                &k.to_dtype(dtype)?.to_device(&cuda)?,
                &v.to_dtype(dtype)?.to_device(&cuda)?,
                [3, 3, 3],
                2f32.sqrt().recip(),
            )?;
            cuda.synchronize()?;
            let actual = actual
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(expected, actual)| (expected - actual).abs() < tolerance),
                "{dtype:?} CUDA neighborhood attention exceeded tolerance {tolerance}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rejects_kernel_above_shared_mem_limit() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        // 33^3 * 4 bytes = 143,748 bytes, comfortably above the 48 KiB
        // default dynamic shared memory budget checked before dispatch.
        let tensor = Tensor::ones((1, 33, 33, 33, 1, 1), DType::F32, &cuda)?;
        let error = neighborhood_attention3d(&tensor, &tensor, &tensor, [33, 33, 33], 1.0)
            .expect_err("oversized shared memory allocation must fail before dispatch");
        assert!(
            error.to_string().contains("shared memory"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_cpu_when_kernel_exceeds_grid_extent() -> Result<()> {
        // Exact repro shape from a real LTX-2.5 diffusion-VAE smoke render:
        // the checkpoint's fixed [11, 11, 11] window is wider than a
        // 9-frame clip's time axis, so `time` must clamp to full attention
        // while `height`/`width` keep sliding normally.
        let elements = 9 * 64 * 64 * 2;
        let values: Vec<f32> = (0..elements)
            .map(|index| (index as f32 % 29.0) / 5.0)
            .collect();
        let q = Tensor::from_vec(values.clone(), (1, 9, 64, 64, 1, 2), &Device::Cpu)?;
        let k = Tensor::from_vec(
            values.iter().rev().copied().collect::<Vec<_>>(),
            (1, 9, 64, 64, 1, 2),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(values, (1, 9, 64, 64, 1, 2), &Device::Cpu)?;
        let expected = neighborhood_attention3d(&q, &k, &v, [11, 11, 11], 2f32.sqrt().recip())?;

        let cuda = Device::new_cuda(0)?;
        let actual = neighborhood_attention3d(
            &q.to_device(&cuda)?,
            &k.to_device(&cuda)?,
            &v.to_device(&cuda)?,
            [11, 11, 11],
            2f32.sqrt().recip(),
        )?;
        cuda.synchronize()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-3));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_cpu_when_kernel_exceeds_every_axis() -> Result<()> {
        // Every axis clamps at once, including two EVEN extents (4): time=1
        // clamps kernel_t 3 -> 1, height=4 and width=4 each clamp kernel_h
        // and kernel_w 7 -> 4, so this degrades to full attention over the
        // whole grid regardless of query position.
        let values: Vec<f32> = (0..(4 * 4 * 2))
            .map(|index| (index as f32 % 13.0) / 3.0)
            .collect();
        let q = Tensor::from_vec(values.clone(), (1, 1, 4, 4, 1, 2), &Device::Cpu)?;
        let k = Tensor::from_vec(
            values.iter().rev().copied().collect::<Vec<_>>(),
            (1, 1, 4, 4, 1, 2),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(values, (1, 1, 4, 4, 1, 2), &Device::Cpu)?;
        let expected = neighborhood_attention3d(&q, &k, &v, [3, 7, 7], 2f32.sqrt().recip())?;

        let cuda = Device::new_cuda(0)?;
        let actual = neighborhood_attention3d(
            &q.to_device(&cuda)?,
            &k.to_device(&cuda)?,
            &v.to_device(&cuda)?,
            [3, 7, 7],
            2f32.sqrt().recip(),
        )?;
        cuda.synchronize()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-4));
        Ok(())
    }
}
