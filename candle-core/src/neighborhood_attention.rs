use crate::{CpuStorage, DType, Layout, Result, Shape, Tensor};

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
    if kernel[0] > *time || kernel[1] > *height || kernel[2] > *width {
        crate::bail!(
            "neighborhood attention kernel {:?} exceeds input grid [{time},{height},{width}]",
            kernel
        )
    }
    if *head_dim == 0 || *head_dim > 256 {
        crate::bail!("neighborhood attention head dimension must be in 1..=256")
    }
    Ok([*batch, *time, *height, *width, *heads, *head_dim])
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
        let q = q.as_slice::<f32>()?;
        let k = k.as_slice::<f32>()?;
        let v = v.as_slice::<f32>()?;
        let q = &q[q_layout.start_offset()..];
        let k = &k[k_layout.start_offset()..];
        let v = &v[v_layout.start_offset()..];
        let spatial = time * height * width;
        let neighbors: usize = self.kernel.iter().product();
        let mut output = vec![0f32; batch * spatial * heads * head_dim];
        let mut scores = vec![0f32; neighbors];
        for b in 0..batch {
            for qt in 0..time {
                let start_t = qt
                    .saturating_sub(self.kernel[0] / 2)
                    .min(time - self.kernel[0]);
                for qh in 0..height {
                    let start_h = qh
                        .saturating_sub(self.kernel[1] / 2)
                        .min(height - self.kernel[1]);
                    for qw in 0..width {
                        let start_w = qw
                            .saturating_sub(self.kernel[2] / 2)
                            .min(width - self.kernel[2]);
                        let query_position = (qt * height + qh) * width + qw;
                        for head in 0..heads {
                            let query_base =
                                ((b * spatial + query_position) * heads + head) * head_dim;
                            let mut index = 0;
                            for dt in 0..self.kernel[0] {
                                for dh in 0..self.kernel[1] {
                                    for dw in 0..self.kernel[2] {
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
                                for dt in 0..self.kernel[0] {
                                    for dh in 0..self.kernel[1] {
                                        for dw in 0..self.kernel[2] {
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
            self.kernel,
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
    use crate::Device;

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
        let actual = actual.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-4));
        Ok(())
    }
}
