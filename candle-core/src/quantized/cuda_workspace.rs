//! Scratch allocations share the device's lifetime, never a global device-id
//! cache. Explicit completion events also cover CUDA's per-thread stream handle:
//! the last device clone may be destroyed on a different host thread.

use crate::cuda_backend::WrapErr;
use crate::{CudaDevice, Result};
use cudarc::driver::{sys, CudaEvent, CudaSlice, CudaStream, DevicePtrMut};
use std::sync::{Arc, Mutex, MutexGuard};

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
    completed: Option<CudaEvent>,
}

impl Drop for WorkspaceSlot {
    fn drop(&mut self) {
        if let (Some(slice), Some(completed)) = (&self.slice, &self.completed) {
            let stream = slice.stream();
            // On a different host thread this may be a different CUDA PTDS.
            // Queue the producer wait before CudaSlice queues its free.
            stream.context().record_err(stream.wait(completed));
        }
    }
}

/// Keep this lease through raw kernel submission. MutexGuard is not Send, so
/// completion is recorded on the same host thread that acquired the scratch.
pub(crate) struct WorkspaceGuard<'a> {
    slot: MutexGuard<'a, WorkspaceSlot>,
    stream: Arc<CudaStream>,
}

impl Drop for WorkspaceGuard<'_> {
    fn drop(&mut self) {
        if let Some(completed) = &self.slot.completed {
            self.stream
                .context()
                .record_err(completed.record(&self.stream));
        }
    }
}

pub(crate) fn ensure<'a>(
    workspace: &'a Mutex<WorkspaceSlot>,
    dev: &CudaDevice,
    bytes: usize,
) -> Result<(u64, WorkspaceGuard<'a>)> {
    let mut slot = workspace.lock().unwrap();
    let stream = dev.cuda_stream();
    if let Some(completed) = &slot.completed {
        // Do not rely on cudarc's multi-stream-mode heuristic: different host
        // threads can use the same special per-thread stream handle.
        stream.wait(completed).w()?;
    } else {
        slot.completed = Some(
            stream
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .w()?,
        );
    }
    let bytes = bytes.max(1);
    if slot.cap < bytes {
        let slice = unsafe { dev.alloc::<u8>(bytes)? };
        slot.slice = Some(slice);
        slot.cap = bytes;
    }
    // The returned lease records the actual final use after the FFI launches;
    // a pointer-extraction guard alone would record it before those launches.
    let ptr = slot
        .slice
        .as_mut()
        .expect("workspace allocated")
        .device_ptr_mut(&stream)
        .0;
    Ok((ptr, WorkspaceGuard { slot, stream }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    #[test]
    fn cross_thread_scratch_drop_waits_for_raw_kernel() -> Result<()> {
        exercise_cross_thread(true)
    }

    #[test]
    fn cross_thread_scratch_reuse_waits_for_raw_kernel() -> Result<()> {
        exercise_cross_thread(false)
    }

    fn exercise_cross_thread(drop_last: bool) -> Result<()> {
        let device = CudaDevice::new(0)?;
        let stream = device.cuda_stream();
        // Finish the constructor-owned cuBLASLt workspace allocation so this
        // fixture reports only the quantized scratch lifetime it exercises.
        stream.synchronize().unwrap();
        let context = stream.context().clone();
        let module = context
            .load_module(cudarc::nvrtc::Ptx::from_src(
                r#"
.version 6.0
.target sm_50
.address_size 64
.visible .entry delayed_read(.param .u64 scratch, .param .u64 output) {
    .reg .b64 %rd<6>;
    .reg .b32 %r<2>;
    .reg .pred %p;
    ld.param.u64 %rd0, [scratch];
    ld.param.u64 %rd1, [output];
    mov.u32 %r0, 42;
    st.global.u32 [%rd0], %r0;
    mov.u64 %rd2, %clock64;
wait:
    mov.u64 %rd3, %clock64;
    sub.u64 %rd4, %rd3, %rd2;
    setp.lt.u64 %p, %rd4, 500000000;
    @%p bra wait;
    ld.volatile.global.u32 %r1, [%rd0];
    st.global.u32 [%rd1], %r1;
    ret;
}
.visible .entry overwrite(.param .u64 scratch) {
    .reg .b64 %rd;
    .reg .b32 %r;
    ld.param.u64 %rd, [scratch];
    mov.u32 %r, 0;
    st.global.u32 [%rd], %r;
    ret;
}
"#,
            ))
            .unwrap();
        let function = module.load_function("delayed_read").unwrap();
        let mut output = stream.alloc_zeros::<u32>(1).unwrap();
        {
            let (ptr, _workspace) = ensure(&device.quantized_workspaces.mmq, &device, 64 << 20)?;
            let mut launch = stream.launch_builder(&function);
            launch.arg(&ptr).arg(&mut output);
            unsafe {
                launch
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        }
        // Give the delayed kernel time to start, then destroy the last Candle
        // device on a different host thread's CUDA per-thread stream.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let other_context = context.clone();
        // Keep cudarc's unrelated cuBLASLt workspace alive on this thread;
        // the drop case is specifically proving quantized scratch teardown.
        let retained_blas_lt = device.blas_lt.clone();
        // Retain a clone for the reuse case, avoiding any incidental library
        // destructor synchronization before the second thread overwrites scratch.
        let retained = (!drop_last).then(|| device.clone());
        std::thread::spawn(move || {
            let other = other_context.per_thread_stream();
            if drop_last {
                drop(device);
                other.synchronize().unwrap();
                let replacement = other.alloc_zeros::<u8>(64 << 20).unwrap();
                other.synchronize().unwrap();
                drop(replacement);
            } else {
                let (ptr, _workspace) =
                    ensure(&device.quantized_workspaces.mmq, &device, 64 << 20).unwrap();
                let overwrite = module.load_function("overwrite").unwrap();
                let mut launch = other.launch_builder(&overwrite);
                launch.arg(&ptr);
                unsafe {
                    launch
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
            }
        })
        .join()
        .unwrap();
        drop(retained);
        let observed = stream.clone_dtoh(&output).unwrap();
        drop(retained_blas_lt);
        assert_eq!(observed, vec![42]);
        Ok(())
    }
}
