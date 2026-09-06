#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::{result, sys, CudaContext};
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    Device, Module, Tensor,
};

fn pool_used(context: &CudaContext) -> u64 {
    context.synchronize().unwrap();
    context.bind_to_thread().unwrap();
    let mut bytes = 0u64;
    // SAFETY: the retained context owns this pool; the attribute writes a u64.
    unsafe {
        let pool = result::device::get_mem_pool(context.cu_device()).unwrap();
        result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&mut bytes as *mut u64).cast(),
        )
        .unwrap();
    }
    bytes
}

#[test]
fn quantized_scratch_released_after_last_device_clone() -> candle_core::Result<()> {
    let context = CudaContext::new(0).unwrap();
    assert!(
        context.has_async_alloc(),
        "test requires CUDA async allocation accounting"
    );
    let baseline = pool_used(&context);
    for dtype in [GgmlDType::Q5K, GgmlDType::Q8_0] {
        for batch in [1, 32] {
            for _ in 0..3 {
                let device = Device::new_cuda(0)?;
                let clone = device.clone();
                exercise(&device, dtype, batch)?;
                drop(device);
                assert!(
                    pool_used(&context) > baseline,
                    "scratch must survive a live device clone"
                );
                exercise(&clone, dtype, batch * 2)?;
                drop(clone);
                assert_eq!(
                    pool_used(&context),
                    baseline,
                    "scratch leaked for {dtype:?}, batch {batch}"
                );
            }
        }
    }
    // Independently constructed streams must not share scratch ownership.
    let first = Device::Cuda(candle_core::CudaDevice::new_with_stream(0)?);
    let second = Device::Cuda(candle_core::CudaDevice::new_with_stream(0)?);
    exercise(&first, GgmlDType::Q5K, 32)?;
    exercise(&second, GgmlDType::Q8_0, 32)?;
    let both = pool_used(&context);
    drop(first);
    let remaining = pool_used(&context);
    assert!(remaining > baseline && remaining < both);
    exercise(&second, GgmlDType::Q8_0, 64)?;
    drop(second);
    assert_eq!(pool_used(&context), baseline);
    Ok(())
}

fn exercise(device: &Device, dtype: GgmlDType, batch: usize) -> candle_core::Result<()> {
    let data: Vec<f32> = (0..256 * 256).map(|i| (i % 17) as f32 / 17.).collect();
    let weights = Tensor::from_vec(data, (256, 256), &Device::Cpu)?;
    let matmul = QMatMul::from_qtensor(QTensor::quantize_to_device(&weights, dtype, device)?)?;
    let input = Tensor::ones((batch, 256), candle_core::DType::F32, device)?;
    let values = matmul.forward(&input)?.to_vec2::<f32>()?;
    let reference = QMatMul::from_qtensor(QTensor::quantize(&weights, dtype)?)?
        .forward(&input.to_device(&Device::Cpu)?)?
        .to_vec2::<f32>()?;
    assert!(
        values
            .iter()
            .flatten()
            .zip(reference.iter().flatten())
            .all(|(v, r)| (v - r).abs() < 2.),
        "CUDA/CPU mismatch for {dtype:?}, batch {batch}"
    );
    Ok(())
}
