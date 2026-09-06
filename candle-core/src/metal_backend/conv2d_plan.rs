//! Spatial partitioning follows cpu_backend/conv2d.rs::conv2d_tiled.
use crate::{Error, Result};
const WORKSPACE_BYTES: usize = 64 << 20;
// Leave room for linear_split's rounded final threadgroup.
const MAX_GRID: usize = u32::MAX as usize - 1023;
pub(super) struct Plan {
    pub rows_per_tile: usize,
    pub output_elements: usize,
}
pub(super) fn plan(rows: usize, patch: usize, channels: usize, bytes: usize) -> Result<Plan> {
    let error = || Error::Msg("Metal conv2d invalid or overflowing workspace dimensions".into());
    if rows == 0 || patch == 0 || channels == 0 || bytes == 0 {
        return Err(error());
    }
    let widest = patch.max(channels);
    let row_bytes = widest.checked_mul(bytes).ok_or_else(error)?;
    let rows_per_tile = (WORKSPACE_BYTES / row_bytes)
        .min(MAX_GRID / widest)
        .min(rows);
    if rows_per_tile == 0 {
        return Err(error());
    }
    // Both the kernel's global coordinate and destination index must fit size_t.
    rows.checked_mul(patch).ok_or_else(error)?;
    let output_elements = rows.checked_mul(channels).ok_or_else(error)?;
    output_elements.checked_mul(bytes).ok_or_else(error)?;
    Ok(Plan {
        rows_per_tile,
        output_elements,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_patch_and_result_workspace() {
        for (patch, channels) in [(2304, 128), (1, 8192)] {
            let p = plan(1 << 24, patch, channels, 4).unwrap();
            assert!(p.rows_per_tile * patch * 4 <= WORKSPACE_BYTES);
            assert!(p.rows_per_tile * channels * 4 <= WORKSPACE_BYTES);
            assert!(p.rows_per_tile * patch <= MAX_GRID);
            assert!(p.rows_per_tile * channels <= MAX_GRID);
        }
    }
    #[test]
    fn large_virtual_grid_uses_small_dispatches() {
        let rows = 1368 * 1368;
        assert!(rows * 2304 > u32::MAX as usize);
        let p = plan(rows, 2304, 128, 4).unwrap();
        assert!(p.rows_per_tile * 2304 < MAX_GRID);
        assert!(rows % p.rows_per_tile != 0);
    }
    #[test]
    fn rejects_zero_and_overflow() {
        for (r, k, n, b) in [
            (0, 1, 1, 4),
            (1, 0, 1, 4),
            (1, 1, 0, 4),
            (1, 1, 1, 0),
            (usize::MAX, 2, 1, 4),
            (2, 1, usize::MAX, 4),
        ] {
            assert!(plan(r, k, n, b).is_err());
        }
    }
}
