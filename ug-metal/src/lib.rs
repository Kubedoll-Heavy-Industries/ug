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
//! use ug_metal::runtime::Device;
//! let device = Device::new()?;
//! ```

pub mod code_gen;
pub mod runtime;
pub mod utils;

pub use runtime::{Device, Slice};
