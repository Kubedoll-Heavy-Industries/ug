//! Core tensor compiler library providing lazy evaluation, scheduling, and code generation.
//!
//! This crate provides the foundational abstractions for building tensor computation graphs
//! that can be compiled and executed on various backends (CPU, CUDA, Metal).
//!
//! # Key Components
//!
//! - [`LazyBuffer`]: Lazy tensor representation that defers computation until explicitly realized
//! - [`Schedule`]: Compilation and execution planning for tensor operations
//! - [`DType`]: Supported data types (F16, BF16, F32, I32, I64)
//! - [`Device`] and [`Slice`]: Backend-agnostic traits for device memory and execution

pub mod block;
pub mod cache;
pub mod common_tests;
pub mod r#const;
pub mod cpu_code_gen;
pub mod cpu_runtime;
pub mod display;
pub mod dtype;
pub mod error;
pub mod interpreter;
pub mod lang;
pub mod layout;
pub mod lazy_buffer;
pub mod lower;
pub mod lower_op;
pub mod safetensors;
pub mod samples;
pub mod schedule;
pub mod utils;

pub use r#const::Const;
pub use cpu_runtime::{CpuDevice, CpuStorage, CpuStorageRef, CpuStorageRefMut};
pub use dtype::{DType, WithDType};
pub use error::{Error, Result};
pub use layout::{D, Dim, Layout, Shape};
pub use lazy_buffer::LazyBuffer;
pub use schedule::{Schedule, ScheduleItem};

pub trait Slice: std::fmt::Debug {
    type Device: Device<Slice = Self>;

    fn device(&self) -> &Self::Device;
    fn dtype(&self) -> DType;
    fn len(&self) -> usize;
    fn copy_host_to_device<DT: WithDType>(&mut self, src: &[DT]) -> Result<()>;
    fn copy_device_to_host<DT: WithDType>(&self, dst: &mut [DT]) -> Result<()>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn to_vec<DT: WithDType>(&self) -> Result<Vec<DT>> {
        let mut host = vec![DT::zero(); self.len()];
        self.copy_device_to_host(&mut host)?;
        Ok(host)
    }
}

pub trait Device: Clone + std::fmt::Debug {
    type Slice: Slice<Device = Self>;
    type Func;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn allocate_uninit(&self, dtype: DType, len: usize) -> Result<Self::Slice>;
    fn synchronize(&self) -> Result<()>;
    fn compile(&self, kernel: &crate::lang::ssa::Kernel, name: Option<&str>) -> Result<Self::Func>;
    // TODO: currently const parameters are hardcoded in the kernel and new code is generated for
    // these when necessary. Maybe we should have a more generic arg type that could handle
    // `Const` scalars.
    fn run(&self, f: &Self::Func, args: &mut [&mut Self::Slice]) -> Result<()>;

    fn matmul(
        &self,
        _dst: &mut Self::Slice,
        _lhs: &Self::Slice,
        _rhs: &Self::Slice,
        _bmnk: (usize, usize, usize, usize),
        _lhs_l: &Layout,
        _rhs_l: &Layout,
    ) -> Result<()>;

    fn use_grid() -> bool;
}
