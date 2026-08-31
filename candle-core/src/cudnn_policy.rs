//! Runtime control over the cuDNN convolution path.
//!
//! Whether cuDNN is *available* is a compile-time question ([`is_compiled`]);
//! whether a given convolution should *use* it is not. Two reasons a caller
//! may want it off in a binary that compiled it in:
//!
//! * **Reproducibility.** cuDNN and the im2col path sum in a different order,
//!   so they do not agree bit-for-bit. An application that promises a stored
//!   seed re-renders the same bytes forever can afford that for some model
//!   families and not others, and that is a per-family decision the backend
//!   cannot make for it.
//! * **Bisecting.** A single switch turns the whole path off at runtime,
//!   which is a far cheaper way to answer "is cuDNN doing this?" than a
//!   rebuild.
//!
//! The switch is global and is read on every convolution. That is sound for
//! the intended use — a host that runs one model at a time and sets the policy
//! before the model runs — and deliberately not a per-tensor knob.
//!
//! With the `cudnn` feature off, [`set_enabled`] is a no-op and [`is_enabled`]
//! is always `false`, so callers can wire the policy unconditionally.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(true);
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

/// Whether this build has the cuDNN convolution path compiled in at all.
pub fn is_compiled() -> bool {
    cfg!(feature = "cudnn")
}

/// Whether convolutions should currently take the cuDNN path.
///
/// Always `false` in a build without the `cudnn` feature, whatever
/// [`set_enabled`] was called with.
pub fn is_enabled() -> bool {
    is_compiled() && ENABLED.load(Ordering::Relaxed)
}

/// Turn the cuDNN convolution path on or off for subsequent convolutions.
///
/// Returns the previous setting, so a caller can restore it.
pub fn set_enabled(enabled: bool) -> bool {
    ENABLED.swap(enabled, Ordering::Relaxed)
}

/// How many convolutions have actually executed on cuDNN in this process.
///
/// The dispatch is conditional twice over — the feature must be compiled, the
/// policy enabled, and the shape has to clear the size thresholds — so "I
/// enabled cuDNN" is not evidence that any convolution took it. A parity test
/// that does not assert on this counter silently compares im2col to itself and
/// passes no matter how wrong the cuDNN path is.
pub fn dispatch_count() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// Record that a convolution executed on cuDNN. Called by the CUDA backend.
pub(crate) fn record_dispatch() {
    DISPATCHED.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_enabled_tracks_the_switch_only_when_compiled() {
        let prev = set_enabled(true);
        assert_eq!(is_enabled(), is_compiled());
        set_enabled(false);
        assert!(!is_enabled());
        // Restoring is part of the contract: `set_enabled` reports what it replaced.
        assert!(!set_enabled(prev));
        assert_eq!(is_enabled(), is_compiled() && prev);
        set_enabled(prev);
    }

    #[test]
    fn a_build_without_the_feature_can_never_enable_it() {
        if !is_compiled() {
            set_enabled(true);
            assert!(!is_enabled());
        }
    }
}
