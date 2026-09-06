//! Scratch allocations share the device's stream and lifetime, never a global
//! device-id cache. CudaSlice owns only the cudarc stream/context, so these slots
//! do not keep their owning Candle device alive.

use crate::{CudaDevice, Result};
use cudarc::driver::{CudaSlice, DevicePtr};
use std::sync::{Mutex, MutexGuard};

#[derive(Default)]
pub(crate) struct QuantizedWorkspaces {
    pub mmq: Mutex<WorkspaceSlot>,
    pub fixup: Mutex<WorkspaceSlot>,
    pub mmvq: Mutex<WorkspaceSlot>,
}

#[derive(Default)]
pub(crate) struct WorkspaceSlot {
    slice: Option<CudaSlice<u8>>,
    cap: usize,
}

/// Hold the returned guard through kernel submission. Resizes and destruction
/// use the same stream's ordered free, so previous launches remain valid.
pub(crate) fn ensure<'a>(
    workspace: &'a Mutex<WorkspaceSlot>,
    dev: &CudaDevice,
    bytes: usize,
) -> Result<(u64, MutexGuard<'a, WorkspaceSlot>)> {
    let mut slot = workspace.lock().unwrap();
    let bytes = bytes.max(1);
    if slot.cap < bytes {
        let slice = unsafe { dev.alloc::<u8>(bytes)? };
        slot.slice = Some(slice);
        slot.cap = bytes;
    }
    let slice = slot.slice.as_ref().expect("workspace allocated");
    let ptr = slice.device_ptr(slice.stream()).0;
    Ok((ptr, slot))
}
