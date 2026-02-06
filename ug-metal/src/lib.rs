//! Metal backend for the ug tensor compiler.
//!
//! This crate provides Apple GPU acceleration via Metal, including:
//! - Metal shader code generation from ug's SSA IR
//! - Device memory management and data transfer
//! - MPS (Metal Performance Shaders) integration for optimized operations
//!
//! # Example
//!
//! ```ignore
//! use ug_metal::MetalDevice;
//! let device = MetalDevice::new()?;
//! ```

pub mod code_gen;
pub mod runtime;
pub mod utils;

// Primary exports with explicit names
pub use runtime::{MetalDevice, MetalSlice};

// Backward compatibility aliases
pub use runtime::{Device, Slice};
