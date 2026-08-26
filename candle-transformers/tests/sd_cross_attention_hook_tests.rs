//! Cross-attention conditioning hooks on the Stable Diffusion UNet.

use candle::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::stable_diffusion::attention::{
    CrossAttentionHook, NoopCrossAttentionHook,
};
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};
use std::cell::RefCell;

const BATCH: usize = 2;
const LATENT: usize = 8;
const TEXT_TOKENS: usize = 3;
const IN_CHANNELS: usize = 4;
const OUT_CHANNELS: usize = 4;

/// A three-level UNet shaped like SDXL: no cross-attention at the top level,
/// several transformers per block deeper down, and a cross-attended mid block.
///
/// `DownBlock2D`/`UpBlock2D` hard-code 32 resnet norm groups, so the channel
/// counts have to stay multiples of 32 however small the latent is.
fn tiny_config() -> UNet2DConditionModelConfig {
    UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig {
                out_channels: 32,
                use_cross_attn: None,
                attention_head_dim: 2,
            },
            BlockConfig {
                out_channels: 64,
                use_cross_attn: Some(2),
                attention_head_dim: 2,
            },
            BlockConfig {
                out_channels: 128,
                use_cross_attn: Some(1),
                attention_head_dim: 4,
            },
        ],
        layers_per_block: 1,
        norm_num_groups: 32,
        cross_attention_dim: 6,
        ..Default::default()
    }
}

/// The `(inner_dim, heads)` of every `attn2` module, in UNet traversal order.
///
/// Derived from the config the way `UNet2DConditionModel::new` builds the
/// blocks: `down_blocks` in order, then the single mid-block transformer, then
/// `up_blocks` over the reversed block list with one extra resnet each.
fn expected_cross_attentions(config: &UNet2DConditionModelConfig) -> Vec<(usize, usize)> {
    fn push(expected: &mut Vec<(usize, usize)>, block: &BlockConfig, resnets: usize) {
        if let Some(transformer_layers) = block.use_cross_attn {
            for _ in 0..resnets * transformer_layers {
                expected.push((block.out_channels, block.attention_head_dim))
            }
        }
    }

    let mut expected = vec![];
    for block in config.blocks.iter() {
        push(&mut expected, block, config.layers_per_block);
    }
    // The mid block has a single transformer, and it is always cross-attended
    // even when the deepest down/up block is not.
    let last = config.blocks.last().expect("at least one block");
    for _ in 0..last.use_cross_attn.unwrap_or(1) {
        expected.push((last.out_channels, last.attention_head_dim))
    }
    for block in config.blocks.iter().rev() {
        push(&mut expected, block, config.layers_per_block + 1);
    }
    expected
}

struct Inputs {
    xs: Tensor,
    encoder_hidden_states: Tensor,
}

fn inputs(config: &UNet2DConditionModelConfig, dev: &Device) -> Result<Inputs> {
    let xs = Tensor::arange(0f32, (BATCH * IN_CHANNELS * LATENT * LATENT) as f32, dev)?
        .reshape((BATCH, IN_CHANNELS, LATENT, LATENT))?
        .affine(0.003, -0.4)?;
    let encoder_hidden_states = Tensor::arange(
        0f32,
        (BATCH * TEXT_TOKENS * config.cross_attention_dim) as f32,
        dev,
    )?
    .reshape((BATCH, TEXT_TOKENS, config.cross_attention_dim))?
    .affine(0.02, -0.3)?;
    Ok(Inputs {
        xs,
        encoder_hidden_states,
    })
}

fn tiny_unet(config: &UNet2DConditionModelConfig, dev: &Device) -> Result<UNet2DConditionModel> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    UNet2DConditionModel::new(vs, IN_CHANNELS, OUT_CHANNELS, false, config.clone())
}

fn fixture() -> Result<(UNet2DConditionModelConfig, UNet2DConditionModel, Inputs)> {
    let dev = Device::Cpu;
    let config = tiny_config();
    let inputs = inputs(&config, &dev)?;
    let unet = tiny_unet(&config, &dev)?;
    Ok((config, unet, inputs))
}

#[test]
fn released_configs_keep_their_cross_attention_counts() {
    // The default block list is the SD1.5 shape.
    assert_eq!(
        expected_cross_attentions(&UNet2DConditionModelConfig::default()).len(),
        16
    );
    let sdxl = UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig {
                out_channels: 320,
                use_cross_attn: None,
                attention_head_dim: 5,
            },
            BlockConfig {
                out_channels: 640,
                use_cross_attn: Some(2),
                attention_head_dim: 10,
            },
            BlockConfig {
                out_channels: 1280,
                use_cross_attn: Some(10),
                attention_head_dim: 20,
            },
        ],
        ..Default::default()
    };
    assert_eq!(expected_cross_attentions(&sdxl).len(), 70);
}

#[test]
fn noop_hook_is_bit_identical_to_plain_forward() -> Result<()> {
    let (_config, unet, i) = fixture()?;
    let plain = unet.forward(&i.xs, 0.7, &i.encoder_hidden_states)?;
    assert_eq!(plain.dims(), &[BATCH, OUT_CHANNELS, LATENT, LATENT]);
    let hooked = unet.forward_with_hook(
        &i.xs,
        0.7,
        &i.encoder_hidden_states,
        &NoopCrossAttentionHook,
    )?;
    assert_eq!(plain.shape(), hooked.shape());
    assert_eq!(
        plain.flatten_all()?.to_vec1::<f32>()?,
        hooked.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}

#[derive(Debug, PartialEq)]
struct Call {
    index: usize,
    query: Vec<usize>,
    attended: Vec<usize>,
    heads: usize,
}

#[derive(Default)]
struct Recorder(RefCell<Vec<Call>>);

impl CrossAttentionHook for Recorder {
    fn cross_attention(
        &self,
        index: usize,
        query: &Tensor,
        attended: &Tensor,
        heads: usize,
    ) -> Result<Option<Tensor>> {
        self.0.borrow_mut().push(Call {
            index,
            query: query.dims().to_vec(),
            attended: attended.dims().to_vec(),
            heads,
        });
        Ok(None)
    }
}

#[test]
fn hook_visits_every_cross_attention_in_traversal_order() -> Result<()> {
    let (config, unet, i) = fixture()?;
    let recorder = Recorder::default();
    let plain = unet.forward(&i.xs, 0.7, &i.encoder_hidden_states)?;
    let hooked = unet.forward_with_hook(&i.xs, 0.7, &i.encoder_hidden_states, &recorder)?;
    assert_eq!(
        plain.flatten_all()?.to_vec1::<f32>()?,
        hooked.flatten_all()?.to_vec1::<f32>()?,
        "an observing hook must not change the output"
    );

    let expected = expected_cross_attentions(&config);
    assert!(expected.len() > 1, "the tiny config must exercise several");
    let calls = recorder.0.borrow();
    assert_eq!(calls.len(), expected.len());
    for (position, (call, &(inner_dim, heads))) in calls.iter().zip(expected.iter()).enumerate() {
        assert_eq!(call.index, position, "indices must run 0..N in order");
        assert_eq!(call.heads, heads, "at index {position}");
        assert_eq!(call.query.len(), 3, "at index {position}");
        assert_eq!(call.query[0], BATCH, "at index {position}");
        assert_eq!(call.query[2], inner_dim, "at index {position}");
        assert_eq!(call.query, call.attended, "at index {position}");
    }
    Ok(())
}

/// Replaces every attention output with `attended + 1`, counting the calls.
#[derive(Default)]
struct BumpAll(RefCell<usize>);

impl CrossAttentionHook for BumpAll {
    fn cross_attention(
        &self,
        _index: usize,
        _query: &Tensor,
        attended: &Tensor,
        _heads: usize,
    ) -> Result<Option<Tensor>> {
        *self.0.borrow_mut() += 1;
        Ok(Some((attended + 1.0)?))
    }
}

/// Replaces the attention output at one index only.
struct BumpOne(usize);

impl CrossAttentionHook for BumpOne {
    fn cross_attention(
        &self,
        index: usize,
        _query: &Tensor,
        attended: &Tensor,
        _heads: usize,
    ) -> Result<Option<Tensor>> {
        if index == self.0 {
            Ok(Some((attended + 1.0)?))
        } else {
            Ok(None)
        }
    }
}

#[test]
fn replacements_flow_through_to_out() -> Result<()> {
    let (config, unet, i) = fixture()?;
    let plain = unet
        .forward(&i.xs, 0.7, &i.encoder_hidden_states)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let bump_all = BumpAll::default();
    let bumped = unet
        .forward_with_hook(&i.xs, 0.7, &i.encoder_hidden_states, &bump_all)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_ne!(plain, bumped, "the replacement was ignored");

    let expected = expected_cross_attentions(&config);
    assert_eq!(
        *bump_all.0.borrow(),
        expected.len(),
        "replacing must not change how many attn2 modules are visited"
    );

    for index in 0..expected.len() {
        let bumped = unet
            .forward_with_hook(&i.xs, 0.7, &i.encoder_hidden_states, &BumpOne(index))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_ne!(plain, bumped, "replacement at index {index} was ignored");
    }
    Ok(())
}

/// Returns an attention output with one row of the sequence dropped.
struct TruncatingHook;

impl CrossAttentionHook for TruncatingHook {
    fn cross_attention(
        &self,
        _index: usize,
        _query: &Tensor,
        attended: &Tensor,
        _heads: usize,
    ) -> Result<Option<Tensor>> {
        Ok(Some(attended.narrow(1, 0, attended.dim(1)? - 1)?))
    }
}

/// Returns the attention output in a different dtype.
struct RetypingHook;

impl CrossAttentionHook for RetypingHook {
    fn cross_attention(
        &self,
        _index: usize,
        _query: &Tensor,
        attended: &Tensor,
        _heads: usize,
    ) -> Result<Option<Tensor>> {
        Ok(Some(attended.to_dtype(DType::F16)?))
    }
}

#[test]
fn malformed_replacements_are_rejected() -> Result<()> {
    let (_config, unet, i) = fixture()?;
    let err = unet
        .forward_with_hook(&i.xs, 0.7, &i.encoder_hidden_states, &TruncatingHook)
        .expect_err("shape change must fail");
    assert!(err.to_string().contains("cross-attention hook"), "{err}");
    assert!(err.to_string().contains("index 0"), "{err}");

    let err = unet
        .forward_with_hook(&i.xs, 0.7, &i.encoder_hidden_states, &RetypingHook)
        .expect_err("dtype change must fail");
    assert!(err.to_string().contains("cross-attention hook"), "{err}");
    Ok(())
}
