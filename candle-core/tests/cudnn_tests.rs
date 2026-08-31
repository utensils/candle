//! The cuDNN convolution path against the im2col path it replaces.
//!
//! Before this file candle had no test coverage for the `cudnn` feature at all
//! — it appeared only in READMEs — so a wrong descriptor, a wrong math type or
//! a bad dispatch decision produced silently wrong pixels.
//!
//! Every comparison here asserts, via [`cudnn_policy::dispatch_count`], that a
//! convolution *actually executed on cuDNN*. That assertion is the point of
//! the file: dispatch is conditional three times over (feature compiled,
//! policy enabled, shape past the size threshold), so a suite that only sets
//! the policy and compares outputs will happily run im2col twice and pass no
//! matter how broken cuDNN is. An earlier version of this file did exactly
//! that — mutating the cuDNN padding left all eight tests green.
//!
//! The two paths sum in a different order and never agree bit-for-bit; the
//! tolerances are set from measured deviation on an RTX 4090 (cuDNN 9.13).

use candle_core::{cudnn_policy, DType, Device, IndexOp, Result, Tensor};

/// A CUDA device, but only in a build that can actually dispatch to cuDNN.
///
/// `cuda` without `cudnn` is a supported configuration: the device opens, the
/// convolutions run, and `is_enabled()` is permanently false. Every comparison
/// below would then observe zero dispatches and fail for a reason that is not
/// a defect, so the whole suite opts out instead.
fn cudnn_device() -> Option<Device> {
    if !cudnn_policy::is_compiled() {
        return None;
    }
    Device::new_cuda(0).ok()
}

/// Whether this GPU can run `dtype` convolutions on cuDNN at all.
///
/// bf16 convolution needs SM80+; on an older card cuDNN cannot execute it, the
/// backend deliberately falls back to im2col, and demanding a dispatch would
/// fail the suite on hardware it is supposed to support. Probing with a shape
/// known to clear the threshold separates "this card cannot" from "the
/// dispatch decision is wrong" — the latter must still fail, so the per-shape
/// dispatch assertions below stay exactly as strict.
fn cudnn_runs_dtype(dev: &Device, dtype: DType) -> Result<bool> {
    let (c_in, c_out, h, w, k, pad) = DISPATCHING_SHAPES[0];
    let x = Tensor::randn(0f32, 1.0, (1, c_in, h, w), dev)?.to_dtype(dtype)?;
    let wt = Tensor::randn(0f32, 0.05, (c_out, c_in, k, k), dev)?.to_dtype(dtype)?;
    let prev = cudnn_policy::set_enabled(true);
    let before = cudnn_policy::dispatch_count();
    let _ = x.conv2d(&wt, pad, 1, 1, 1)?;
    let dispatched = cudnn_policy::dispatch_count() - before;
    cudnn_policy::set_enabled(prev);
    Ok(dispatched > 0)
}

/// Run `f` with cuDNN forced on, then forced off, and hand back both results.
/// Fails if the first run never reached cuDNN.
fn both_paths<T>(what: &str, f: impl Fn() -> Result<T>) -> Result<(T, T)> {
    let prev = cudnn_policy::set_enabled(true);
    let before = cudnn_policy::dispatch_count();
    let with = f();
    let dispatched = cudnn_policy::dispatch_count() - before;
    cudnn_policy::set_enabled(false);
    let without = f();
    cudnn_policy::set_enabled(prev);
    assert!(
        dispatched > 0,
        "{what}: nothing reached cuDNN, so this comparison proves nothing — \
         the shape is below the dispatch threshold or the path errored out"
    );
    Ok((with?, without?))
}

fn relative_deviation(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let scale = b
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?
        .max(1e-6);
    let dev = (a - &b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(dev / scale)
}

/// (c_in, c_out, h, w, k, pad). Every entry clears the *half-precision*
/// dispatch threshold, the higher of the two, so one list exercises cuDNN in
/// f32, f16 and bf16 alike.
const DISPATCHING_SHAPES: &[(usize, usize, usize, usize, usize, usize)] = &[
    (64, 96, 240, 416, 3, 1),   // the Wan VAE's dominant decoder shape
    (192, 192, 120, 208, 3, 1), // narrow spatial, wide channels
    (24, 48, 300, 400, 5, 2),   // odd extents, 5x5 kernel
    (16, 32, 480, 832, 3, 0),   // no padding, few channels, large canvas
];

fn conv2d_paths_agree(dev: &Device, dtype: DType, tol: f32) -> Result<()> {
    if !cudnn_runs_dtype(dev, dtype)? {
        eprintln!("skipping {dtype:?}: this GPU's cuDNN cannot execute it");
        return Ok(());
    }
    for &(c_in, c_out, h, w, k, pad) in DISPATCHING_SHAPES {
        let x = Tensor::randn(0f32, 1.0, (1, c_in, h, w), dev)?.to_dtype(dtype)?;
        let wt = Tensor::randn(0f32, 0.05, (c_out, c_in, k, k), dev)?.to_dtype(dtype)?;
        let label = format!("{c_in}->{c_out} {k}x{k} @{h}x{w} {dtype:?}");
        let (a, b) = both_paths(&label, || x.conv2d(&wt, pad, 1, 1, 1))?;
        assert_eq!(a.dims(), b.dims(), "{label}: shape disagreement");
        let deviation = relative_deviation(&a, &b)?;
        assert!(
            deviation <= tol,
            "{label}: cuDNN and im2col disagree, relative deviation {deviation:e} exceeds {tol:e}"
        );
    }
    Ok(())
}

#[test]
fn cudnn_and_im2col_agree_f32() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    // FMA math forbids TF32, so f32 stays genuinely f32 on both paths. A
    // descriptor left at CUDNN_DEFAULT_MATH lands around 3e-4 and trips this.
    conv2d_paths_agree(&dev, DType::F32, 5e-5)
}

#[test]
fn cudnn_and_im2col_agree_bf16() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    conv2d_paths_agree(&dev, DType::BF16, 2e-2)
}

#[test]
fn cudnn_and_im2col_agree_f16() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    conv2d_paths_agree(&dev, DType::F16, 2e-2)
}

#[test]
fn cudnn_and_im2col_agree_on_strided_input() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    // A non-contiguous source takes cudnn's `create_4d_tensor_ex` stride path,
    // which is the arm most likely to be wired up wrong.
    // Narrowing only the batch dim would NOT do this: candle ignores strides
    // on length-1 dimensions, so the result is still `is_contiguous()` and
    // silently takes the ordinary descriptor. Narrowing width leaves the row
    // stride at the original 416 and is genuinely non-contiguous.
    let x = Tensor::randn(0f32, 1.0, (1, 64, 240, 416), &dev)?.to_dtype(DType::F32)?;
    let x = x.narrow(3, 8, 400)?;
    assert!(
        !x.is_contiguous(),
        "this test is pointless unless the input is actually strided"
    );
    let wt = Tensor::randn(0f32, 0.05, (96, 64, 3, 3), &dev)?.to_dtype(DType::F32)?;
    let (a, b) = both_paths("strided input", || x.conv2d(&wt, 1, 1, 1, 1))?;
    assert!(
        relative_deviation(&a, &b)? <= 5e-5,
        "strided-input paths disagree"
    );
    Ok(())
}

#[test]
fn cudnn_and_im2col_agree_with_stride_and_dilation() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    let x = Tensor::randn(0f32, 1.0, (1, 128, 480, 832), &dev)?.to_dtype(DType::F32)?;
    let wt = Tensor::randn(0f32, 0.05, (128, 128, 3, 3), &dev)?.to_dtype(DType::F32)?;
    for (stride, dilation) in [(2usize, 1usize), (1, 2), (2, 2)] {
        let what = format!("stride {stride} dilation {dilation}");
        let (a, b) = both_paths(&what, || x.conv2d(&wt, 1, stride, dilation, 1))?;
        assert_eq!(a.dims(), b.dims(), "{what}: shape disagreement");
        assert!(
            relative_deviation(&a, &b)? <= 5e-5,
            "{what}: paths disagree"
        );
    }
    Ok(())
}

#[test]
fn grouped_convolutions_agree() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    // candle chunks groups *above* the backend, so the cuDNN path only ever
    // sees groups == 1 — `launch_conv2d` never calls `set_group_count` and
    // would compute the wrong thing if it did. This pins that assumption.
    let x = Tensor::randn(0f32, 1.0, (1, 256, 240, 416), &dev)?.to_dtype(DType::F32)?;
    let wt = Tensor::randn(0f32, 0.05, (256, 64, 3, 3), &dev)?.to_dtype(DType::F32)?;
    let (a, b) = both_paths("grouped", || x.conv2d(&wt, 1, 1, 1, 4))?;
    assert!(
        relative_deviation(&a, &b)? <= 5e-5,
        "grouped conv paths disagree"
    );
    Ok(())
}

#[test]
fn conv1d_paths_agree() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    let x = Tensor::randn(0f32, 1.0, (1, 128, 40_000), &dev)?.to_dtype(DType::F32)?;
    let wt = Tensor::randn(0f32, 0.05, (128, 128, 5), &dev)?.to_dtype(DType::F32)?;
    let (a, b) = both_paths("conv1d", || x.conv1d(&wt, 2, 1, 1, 1))?;
    assert_eq!(a.dims(), b.dims());
    assert!(relative_deviation(&a, &b)? <= 5e-5, "conv1d paths disagree");
    Ok(())
}

#[test]
fn shapes_below_the_threshold_stay_on_im2col() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    let prev = cudnn_policy::set_enabled(true);
    // A 1x1 convolution is a matmul: im2col's column buffer is a free reshape,
    // so cuDNN can only add per-call setup. And a small convolution finishes
    // in less time than cuDNN spends creating descriptors.
    // A named alias: the bare nested tuple trips `clippy::type_complexity`,
    // and this crate's CI runs clippy over tests with `-D warnings`.
    type BelowThresholdCase = (
        &'static str,
        (usize, usize, usize, usize),
        (usize, usize, usize, usize),
        usize,
    );
    let cases: &[BelowThresholdCase] = &[
        (
            "1x1 pointwise, large",
            (1, 256, 240, 416),
            (256, 256, 1, 1),
            0,
        ),
        ("3x3 but tiny canvas", (1, 32, 30, 52), (32, 32, 3, 3), 1),
    ];
    for (what, xs, ws, pad) in cases {
        let x = Tensor::randn(0f32, 1.0, *xs, &dev)?.to_dtype(DType::F32)?;
        let wt = Tensor::randn(0f32, 0.05, *ws, &dev)?.to_dtype(DType::F32)?;
        let before = cudnn_policy::dispatch_count();
        let _ = x.conv2d(&wt, *pad, 1, 1, 1)?;
        assert_eq!(
            cudnn_policy::dispatch_count(),
            before,
            "{what}: took cuDNN, but it is measurably slower than im2col there"
        );
    }
    cudnn_policy::set_enabled(prev);
    Ok(())
}

#[test]
fn disabling_the_policy_routes_everything_to_im2col() -> Result<()> {
    let Some(dev) = cudnn_device() else {
        return Ok(());
    };
    let x = Tensor::ones((1, 64, 240, 416), DType::F32, &dev)?;
    let wt = Tensor::ones((96, 64, 3, 3), DType::F32, &dev)?;
    let prev = cudnn_policy::set_enabled(false);
    let before = cudnn_policy::dispatch_count();
    let y = x.conv2d(&wt, 0, 1, 1, 1)?;
    let dispatched = cudnn_policy::dispatch_count() - before;
    cudnn_policy::set_enabled(prev);
    assert_eq!(
        dispatched, 0,
        "policy was off but a convolution still took cuDNN"
    );
    // Interior of an all-ones convolution: 64 channels x 3 x 3 taps.
    let interior = y.i((0, 0, 10, 10))?.to_scalar::<f32>()?;
    assert!(
        (interior - 576.0).abs() < 1e-3,
        "expected 576.0, got {interior}"
    );
    Ok(())
}
