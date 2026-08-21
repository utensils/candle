//! Post-block conditioning hooks on the dense and quantized FLUX transformers.

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::flux::model::{Config, Flux};
use candle_transformers::models::flux::quantized_model::Flux as QuantizedFlux;
use candle_transformers::models::flux::{BlockHook, NoopBlockHook, WithForward};
use candle_transformers::quantized_var_builder::VarBuilder as QuantizedVarBuilder;
use std::cell::RefCell;

const IMG_TOKENS: usize = 6;
const TXT_TOKENS: usize = 4;

fn tiny_config() -> Config {
    Config {
        in_channels: 8,
        vec_in_dim: 10,
        context_in_dim: 12,
        hidden_size: 32,
        mlp_ratio: 2.0,
        num_heads: 2,
        depth: 2,
        depth_single_blocks: 3,
        axes_dim: vec![4, 6, 6],
        theta: 10_000,
        qkv_bias: true,
        guidance_embed: true,
    }
}

struct Inputs {
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor,
}

fn inputs(cfg: &Config, dev: &Device) -> Result<Inputs> {
    let img = Tensor::arange(0f32, (IMG_TOKENS * cfg.in_channels) as f32, dev)?
        .reshape((1, IMG_TOKENS, cfg.in_channels))?
        .affine(0.01, -0.2)?;
    let img_ids =
        Tensor::arange(0f32, (IMG_TOKENS * 3) as f32, dev)?.reshape((1, IMG_TOKENS, 3))?;
    let txt = Tensor::arange(0f32, (TXT_TOKENS * cfg.context_in_dim) as f32, dev)?
        .reshape((1, TXT_TOKENS, cfg.context_in_dim))?
        .affine(0.02, -0.5)?;
    let txt_ids = Tensor::zeros((1, TXT_TOKENS, 3), DType::F32, dev)?;
    Ok(Inputs {
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps: Tensor::new(&[0.5f32], dev)?,
        y: Tensor::arange(0f32, cfg.vec_in_dim as f32, dev)?.reshape((1, cfg.vec_in_dim))?,
        guidance: Tensor::new(&[3.5f32], dev)?,
    })
}

/// Abstracts over the two transformer ports so every test runs against both.
trait HookedModel {
    fn plain(&self, i: &Inputs) -> Result<Tensor>;
    fn hooked(&self, i: &Inputs, hook: &dyn BlockHook) -> Result<Tensor>;
}

impl HookedModel for Flux {
    fn plain(&self, i: &Inputs) -> Result<Tensor> {
        self.forward(
            &i.img,
            &i.img_ids,
            &i.txt,
            &i.txt_ids,
            &i.timesteps,
            &i.y,
            Some(&i.guidance),
        )
    }
    fn hooked(&self, i: &Inputs, hook: &dyn BlockHook) -> Result<Tensor> {
        self.forward_with_hook(
            &i.img,
            &i.img_ids,
            &i.txt,
            &i.txt_ids,
            &i.timesteps,
            &i.y,
            Some(&i.guidance),
            hook,
        )
    }
}

impl HookedModel for QuantizedFlux {
    fn plain(&self, i: &Inputs) -> Result<Tensor> {
        self.forward(
            &i.img,
            &i.img_ids,
            &i.txt,
            &i.txt_ids,
            &i.timesteps,
            &i.y,
            Some(&i.guidance),
        )
    }
    fn hooked(&self, i: &Inputs, hook: &dyn BlockHook) -> Result<Tensor> {
        self.forward_with_hook(
            &i.img,
            &i.img_ids,
            &i.txt,
            &i.txt_ids,
            &i.timesteps,
            &i.y,
            Some(&i.guidance),
            hook,
        )
    }
}

/// Builds the dense model from random-initialised vars, then re-serialises
/// those exact vars as F32 GGUF tensors so the quantized port shares them.
fn models(cfg: &Config, dev: &Device) -> Result<(Flux, QuantizedFlux)> {
    let varmap = VarMap::new();
    let dense = Flux::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev))?;

    let vars = varmap.data().lock().unwrap();
    let mut qtensors = Vec::with_capacity(vars.len());
    for (name, var) in vars.iter() {
        qtensors.push((
            name.clone(),
            QTensor::quantize(var.as_tensor(), GgmlDType::F32)?,
        ));
    }
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let mut buf = std::io::Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &[], &refs)?;
    let quantized = QuantizedFlux::new(
        cfg,
        QuantizedVarBuilder::from_gguf_buffer(buf.get_ref(), dev)?,
    )?;
    Ok((dense, quantized))
}

fn for_each_model(f: impl Fn(&dyn HookedModel, &Config, &Inputs) -> Result<()>) -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let inputs = inputs(&cfg, &dev)?;
    let (dense, quantized) = models(&cfg, &dev)?;
    f(&dense, &cfg, &inputs)?;
    f(&quantized, &cfg, &inputs)
}

fn assert_bit_identical(a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.to_vec3::<f32>()?, b.to_vec3::<f32>()?);
    Ok(())
}

#[test]
fn noop_hook_is_bit_identical_to_plain_forward() -> Result<()> {
    for_each_model(|model, _cfg, i| {
        let plain = model.plain(i)?;
        assert_eq!(plain.dims(), &[1, IMG_TOKENS, tiny_config().in_channels]);
        assert_bit_identical(&plain, &model.hooked(i, &NoopBlockHook)?)
    })
}

#[derive(Debug, PartialEq)]
enum Event {
    Double {
        index: usize,
        img: Vec<usize>,
        txt: Vec<usize>,
    },
    Single {
        index: usize,
        txt_len: usize,
        xs: Vec<usize>,
    },
}

#[derive(Default)]
struct Recorder(RefCell<Vec<Event>>);

impl BlockHook for Recorder {
    fn after_double_block(
        &self,
        index: usize,
        img: &Tensor,
        txt: &Tensor,
    ) -> Result<Option<Tensor>> {
        self.0.borrow_mut().push(Event::Double {
            index,
            img: img.dims().to_vec(),
            txt: txt.dims().to_vec(),
        });
        Ok(None)
    }
    fn after_single_block(
        &self,
        index: usize,
        txt_len: usize,
        xs: &Tensor,
    ) -> Result<Option<Tensor>> {
        self.0.borrow_mut().push(Event::Single {
            index,
            txt_len,
            xs: xs.dims().to_vec(),
        });
        Ok(None)
    }
}

#[test]
fn hook_observes_every_block_in_order_with_hidden_shapes() -> Result<()> {
    for_each_model(|model, cfg, i| {
        let recorder = Recorder::default();
        assert_bit_identical(&model.plain(i)?, &model.hooked(i, &recorder)?)?;

        let h = cfg.hidden_size;
        let mut expected: Vec<Event> = (0..cfg.depth)
            .map(|index| Event::Double {
                index,
                img: vec![1, IMG_TOKENS, h],
                txt: vec![1, TXT_TOKENS, h],
            })
            .collect();
        expected.extend((0..cfg.depth_single_blocks).map(|index| Event::Single {
            index,
            txt_len: TXT_TOKENS,
            xs: vec![1, TXT_TOKENS + IMG_TOKENS, h],
        }));
        assert_eq!(*recorder.0.borrow(), expected);
        Ok(())
    })
}

/// A per-channel ramp: the one perturbation a LayerNorm or RmsNorm cannot
/// undo, unlike a uniform shift or scale of the hidden vector.
fn channel_ramp(like: &Tensor) -> Result<Tensor> {
    let hidden = like.dim(2)?;
    Tensor::arange(0f32, hidden as f32, like.device())?
        .affine(1.0 / hidden as f64, 0.5)?
        .reshape((1, 1, hidden))?
        .broadcast_as(like.shape())?
        .contiguous()
}

/// Adds a channel ramp to the image stream after one double block.
struct DoubleBump(usize);

impl BlockHook for DoubleBump {
    fn after_double_block(
        &self,
        index: usize,
        img: &Tensor,
        _txt: &Tensor,
    ) -> Result<Option<Tensor>> {
        if index == self.0 {
            Ok(Some((img + channel_ramp(img)?)?))
        } else {
            Ok(None)
        }
    }
}

/// Scales only the image slice by a channel ramp after one single block,
/// leaving the text prefix alone.
struct SingleImageScale(usize);

impl BlockHook for SingleImageScale {
    fn after_single_block(
        &self,
        index: usize,
        txt_len: usize,
        xs: &Tensor,
    ) -> Result<Option<Tensor>> {
        if index != self.0 {
            return Ok(None);
        }
        let txt = xs.narrow(1, 0, txt_len)?;
        let img = xs.narrow(1, txt_len, xs.dim(1)? - txt_len)?;
        let img = (&img * channel_ramp(&img)?)?;
        Ok(Some(Tensor::cat(&[&txt, &img], 1)?))
    }
}

#[test]
fn replacements_reach_the_next_block() -> Result<()> {
    for_each_model(|model, cfg, i| {
        let plain = model.plain(i)?.to_vec3::<f32>()?;
        for index in 0..cfg.depth {
            let bumped = model.hooked(i, &DoubleBump(index))?.to_vec3::<f32>()?;
            assert_ne!(
                plain, bumped,
                "double block {index} replacement was ignored"
            );
        }
        for index in 0..cfg.depth_single_blocks {
            let scaled = model
                .hooked(i, &SingleImageScale(index))?
                .to_vec3::<f32>()?;
            assert_ne!(
                plain, scaled,
                "single block {index} replacement was ignored"
            );
        }
        Ok(())
    })
}

/// Returns the text prefix only, which changes the sequence length.
struct TruncatingHook;

impl BlockHook for TruncatingHook {
    fn after_single_block(
        &self,
        _index: usize,
        txt_len: usize,
        xs: &Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(Some(xs.i((.., ..txt_len))?))
    }
}

/// Returns the image stream in a different dtype.
struct RetypingHook;

impl BlockHook for RetypingHook {
    fn after_double_block(
        &self,
        _index: usize,
        img: &Tensor,
        _txt: &Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(Some(img.to_dtype(DType::F16)?))
    }
}

#[test]
fn malformed_replacements_are_rejected() -> Result<()> {
    for_each_model(|model, _cfg, i| {
        let err = model
            .hooked(i, &TruncatingHook)
            .expect_err("shape change must fail");
        assert!(err.to_string().contains("single block 0"), "{err}");
        let err = model
            .hooked(i, &RetypingHook)
            .expect_err("dtype change must fail");
        assert!(err.to_string().contains("double block 0"), "{err}");
        Ok(())
    })
}
