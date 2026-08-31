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
//! # The policy is per thread, not per process
//!
//! A convolution deep inside a model has no idea which caller invoked it, so
//! the policy has to live somewhere ambient. Making that ambient place a
//! *process* global would be wrong for any host that drives more than one
//! device at once: those hosts run one worker thread per device, and a global
//! would let a thread that wants cuDNN switch it on underneath a thread that
//! is mid-convolution and requires the reproducible path. The bug is silent —
//! the wrong-path convolution still returns a plausible result, just not the
//! bytes the caller promised.
//!
//! Thread-local state is the natural fit: a CUDA op executes inline on the
//! thread that issued it, so "the policy this thread set" and "the policy this
//! convolution runs under" are the same thing by construction. Threads do not
//! inherit it — each new thread starts at the default below — so a worker sets
//! its policy once and cannot be perturbed by any other worker.
//!
//! The default is `false` (im2col). A caller that has expressed no preference
//! gets the path whose bytes do not move.
//!
//! With the `cudnn` feature off, [`set_enabled`] still records the request but
//! [`is_enabled`] is always `false`, so callers can wire the policy
//! unconditionally.

use std::cell::Cell;

thread_local! {
    /// Whether *this* thread's convolutions should take cuDNN.
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    /// How many convolutions have executed on cuDNN on *this* thread.
    static DISPATCHED: Cell<u64> = const { Cell::new(0) };
}

/// Whether this build has the cuDNN convolution path compiled in at all.
pub fn is_compiled() -> bool {
    cfg!(feature = "cudnn")
}

/// Whether convolutions on this thread should currently take the cuDNN path.
///
/// Always `false` in a build without the `cudnn` feature, whatever
/// [`set_enabled`] was called with.
pub fn is_enabled() -> bool {
    is_compiled() && ENABLED.with(Cell::get)
}

/// Turn the cuDNN convolution path on or off for subsequent convolutions
/// **on the calling thread**.
///
/// Returns the previous setting, so a caller can restore it.
pub fn set_enabled(enabled: bool) -> bool {
    ENABLED.with(|cell| cell.replace(enabled))
}

/// How many convolutions have actually executed on cuDNN on this thread.
///
/// The dispatch is conditional several times over — the feature must be
/// compiled, the policy enabled, the shape has to clear the size thresholds,
/// and the launch itself must succeed rather than fall back — so "I enabled
/// cuDNN" is not evidence that any convolution took it. A parity test that
/// does not assert on this counter silently compares im2col to itself and
/// passes no matter how wrong the cuDNN path is.
///
/// Being per-thread is what lets such tests run concurrently: each observes
/// only the convolutions it issued.
pub fn dispatch_count() -> u64 {
    DISPATCHED.with(Cell::get)
}

/// Record that a convolution actually executed on cuDNN.
///
/// Called by the CUDA backend from the innermost launch, after the point where
/// a fallback to another kernel is still possible — counting any earlier would
/// report work cuDNN never did.
#[cfg(feature = "cudnn")]
pub(crate) fn record_dispatch() {
    DISPATCHED.with(|cell| cell.set(cell.get() + 1));
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

    #[test]
    fn the_default_is_the_reproducible_path() {
        // Asserted on a fresh thread: the value on *this* thread may already
        // have been moved by another test in the same binary.
        let default_enabled = std::thread::spawn(is_enabled).join().unwrap();
        assert!(
            !default_enabled,
            "a thread that expressed no preference must get im2col, whose bytes do not move"
        );
    }

    #[test]
    fn one_threads_policy_does_not_leak_into_another() {
        set_enabled(true);
        // The point of the whole module: a second worker is unaffected by what
        // this one just did, so a device rendering stills cannot be switched
        // onto cuDNN by a device rendering clips.
        let other = std::thread::spawn(|| {
            let seen = is_enabled();
            set_enabled(true);
            seen
        })
        .join()
        .unwrap();
        assert!(!other, "cuDNN policy leaked across threads");
        assert_eq!(is_enabled(), is_compiled());
        set_enabled(false);
    }

    #[test]
    fn the_dispatch_counter_is_also_per_thread() {
        // Counting globally would make two concurrent parity tests observe
        // each other's convolutions and compare the wrong paths.
        assert_eq!(std::thread::spawn(dispatch_count).join().unwrap(), 0);
    }
}
