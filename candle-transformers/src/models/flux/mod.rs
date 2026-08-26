//! Flux  Model
//!
//! Flux is a 12B rectified flow transformer capable of generating images from text descriptions.
//!
//! - 🤗 [Hugging Face Model](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
//! - 💻 [GitHub Repository](https://github.com/black-forest-labs/flux)
//! - 📝 [Blog Post](https://blackforestlabs.ai/announcing-black-forest-labs/)
//!
//! # Usage
//!
//! ```bash
//! cargo run --features cuda \
//!     --example flux -r -- \
//!     --height 1024 --width 1024 \
//!     --prompt "a rusty robot walking on a beach holding a small torch, \
//!               the robot has the word \"rust\" written on it, high quality, 4k"
//! ```
//!
//! <div align=center>
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/flux/assets/flux-robot.jpg" alt="" width=320>
//! </div>
//!

use candle::{Result, Tensor};

pub trait WithForward {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor>;
}

/// Observes and optionally rewrites the residual stream after each transformer block.
///
/// Conditioning adapters such as PuLID add identity features at fixed block
/// indices. Implementing this trait lets a caller do that without owning the
/// block lists; every method defaults to "observe only" so a hook can override
/// just the stage it cares about.
pub trait BlockHook {
    /// Runs after double-stream block `index` with that block's output.
    ///
    /// Return `Some(img)` to replace the image stream carried into the next
    /// block. A replacement must keep `img`'s shape and dtype.
    fn after_double_block(
        &self,
        index: usize,
        img: &Tensor,
        txt: &Tensor,
    ) -> Result<Option<Tensor>> {
        let _ = (index, img, txt);
        Ok(None)
    }

    /// Runs after single-stream block `index`.
    ///
    /// `xs` is the `[txt, img]` concatenation along the sequence axis and the
    /// first `txt_len` tokens are the text stream. Return `Some(xs)` to replace
    /// the whole stream; a replacement must keep `xs`'s shape and dtype and
    /// should leave the text prefix untouched.
    fn after_single_block(
        &self,
        index: usize,
        txt_len: usize,
        xs: &Tensor,
    ) -> Result<Option<Tensor>> {
        let _ = (index, txt_len, xs);
        Ok(None)
    }
}

/// The hook used by [`WithForward::forward`]: observes nothing, replaces nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopBlockHook;

impl BlockHook for NoopBlockHook {}

/// Validates a hook's replacement against the tensor it replaces.
fn accept_replacement(
    stage: &str,
    index: usize,
    current: &Tensor,
    replacement: Tensor,
) -> Result<Tensor> {
    if replacement.shape() != current.shape() || replacement.dtype() != current.dtype() {
        candle::bail!(
            "block hook after {stage} block {index} returned {:?} {:?}, expected {:?} {:?}",
            replacement.shape(),
            replacement.dtype(),
            current.shape(),
            current.dtype()
        )
    }
    Ok(replacement)
}

pub mod autoencoder;
pub mod model;
pub mod quantized_model;
pub mod sampling;
