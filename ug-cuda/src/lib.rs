//! CUDA backend for the ug tensor compiler.
//!
//! This crate provides NVIDIA GPU acceleration via CUDA, including:
//! - CUDA kernel code generation from ug's SSA IR
//! - Device memory management and data transfer
//! - cuBLAS integration for optimized matrix multiplication
//!
//! # Example
//!
//! ```ignore
//! use ug_cuda::CudaDevice;
//! let device = CudaDevice::new(0)?;
//! ```

pub use cudarc;
pub mod code_gen;
pub mod gemm;
pub mod runtime;

// Primary exports with explicit names
pub use runtime::{CudaDevice, CudaSlice};

// Re-export cudarc's DeviceSlice for convenience
pub use cudarc::driver::DeviceSlice;

// Backward compatibility aliases
pub use runtime::{Device, Slice};
