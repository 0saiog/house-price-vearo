//! Choosing where tensors live.
//!
//! Vearo's CUDA backend is behind a feature flag, so a build may or may not have
//! a GPU compiled in. The choice is made once, at runtime, and threaded through
//! the trainer and the service rather than hard-coded at each tensor.

use vearo::Device;

/// What the caller asked for on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preference {
    /// Use the GPU when this build has one, otherwise the CPU.
    #[default]
    Auto,
    /// Always the CPU.
    Cpu,
    /// The GPU, or a clear failure if this build has no CUDA backend.
    Cuda,
}

impl std::str::FromStr for Preference {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown device {other:?}, expected auto, cpu or cuda"
            )),
        }
    }
}

/// Registers the backends and resolves `preference` to a device.
///
/// Idempotent, so it is safe to call from a test that also ran the trainer.
///
/// # Panics
/// Panics when `Cuda` is asked for and this build has no CUDA backend. Falling
/// back to the CPU silently would make a benchmark that claims to measure the
/// GPU actually measure the CPU.
#[must_use]
pub fn select(preference: Preference) -> Device {
    vearo::init();
    match preference {
        Preference::Cpu => Device::Cpu,
        Preference::Auto if !vearo::cuda_available() => Device::Cpu,
        Preference::Auto | Preference::Cuda => {
            assert!(
                vearo::cuda_available(),
                "--device cuda needs a CUDA build: rebuild with `--features cuda` \
                 and a CUDA toolkit cudarc accepts (12.5 or older)"
            );
            Device::Cuda(0)
        }
    }
}

/// A short name for logs and reports.
#[must_use]
pub fn name(device: Device) -> &'static str {
    if device.is_cuda() { "cuda" } else { "cpu" }
}
