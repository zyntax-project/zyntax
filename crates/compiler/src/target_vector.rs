//! How wide a vector the target can hold, and how many lanes of a given
//! element that is.
//!
//! Lane count was written as a literal 4 in the vectorizers. That is the
//! right answer for a 128-bit register holding `f32`, and the wrong one
//! for every other combination: `f64` fits two, `i8` fits sixteen, and a
//! 256-bit register fits twice as many of each.
//!
//! **The width is bounded by the narrowest backend that may compile the
//! module, not by what the machine can execute.** One HIR module feeds
//! every tier, and the Cranelift backend accepts 128-bit vectors only,
//! dropping anything else. So a module that any Cranelift tier may
//! compile is capped at 128 regardless of what the host supports. Going
//! wider is sound only where the consumer is known to be LLVM.
//!
//! `ZYNTAX_VECTOR_BITS` overrides the default for experiments. It cannot
//! make a module Cranelift can compile, so it is for measuring the LLVM
//! tier rather than for general use.

use crate::hir::HirType;

/// The widest vector a consumer will accept, in bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorProfile {
    bits: u32,
}

/// Every backend in the tier ladder can take this. Cranelift's vector
/// types are 128-bit and it has no wider form, so this is the width a
/// module compiled by more than one backend must stay within.
pub const PORTABLE_BITS: u32 = 128;

impl Default for VectorProfile {
    fn default() -> Self {
        Self::portable()
    }
}

impl VectorProfile {
    /// The width every backend accepts.
    pub fn portable() -> Self {
        Self {
            bits: PORTABLE_BITS,
        }
    }

    /// An explicit width, rounded down to a power of two and never below
    /// 128, since narrower than one register buys nothing.
    pub fn of_bits(bits: u32) -> Self {
        let bits = bits.max(PORTABLE_BITS);
        Self {
            bits: 1u32 << (31 - bits.leading_zeros()),
        }
    }

    /// What this machine can execute, ignoring which backend will
    /// compile for it. Use only where the consumer is known to accept
    /// the answer.
    pub fn host() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Self { bits: 512 };
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self { bits: 256 };
            }
            Self { bits: 128 }
        }
        // AArch64 NEON registers are 128-bit. SVE is length-agnostic and
        // is not addressed by a fixed lane count, so it is not claimed
        // here.
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self { bits: 128 }
        }
    }

    /// The width the vectorizers should use, honouring an override.
    pub fn effective() -> Self {
        match std::env::var("ZYNTAX_VECTOR_BITS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            Some(bits) => Self::of_bits(bits),
            None => Self::portable(),
        }
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// How many lanes of `elem` fit, or `None` where the element has no
    /// size this can divide or fewer than two would fit.
    pub fn lanes_for(&self, elem: &HirType) -> Option<usize> {
        let elem_bits = match elem {
            HirType::I8 | HirType::U8 | HirType::Bool => 8,
            HirType::I16 | HirType::U16 => 16,
            HirType::I32 | HirType::U32 | HirType::F32 => 32,
            HirType::I64 | HirType::U64 | HirType::F64 => 64,
            _ => return None,
        };
        let lanes = (self.bits / elem_bits) as usize;
        if lanes < 2 {
            None
        } else {
            Some(lanes)
        }
    }
}

/// Lanes of `elem` under the effective profile.
pub fn lanes_for(elem: &HirType) -> Option<usize> {
    VectorProfile::effective().lanes_for(elem)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 128-bit register holds four `f32` and two `f64`. This is the
    /// width that was previously written as a literal 4 for every
    /// element type.
    #[test]
    fn lane_count_follows_the_element_size() {
        let p = VectorProfile::portable();
        assert_eq!(p.lanes_for(&HirType::F32), Some(4));
        assert_eq!(p.lanes_for(&HirType::F64), Some(2));
        assert_eq!(p.lanes_for(&HirType::I32), Some(4));
        assert_eq!(p.lanes_for(&HirType::I8), Some(16));
        assert_eq!(p.lanes_for(&HirType::I16), Some(8));
    }

    /// A wider register holds proportionally more.
    #[test]
    fn a_wider_register_holds_more_lanes() {
        assert_eq!(
            VectorProfile::of_bits(256).lanes_for(&HirType::F32),
            Some(8)
        );
        assert_eq!(
            VectorProfile::of_bits(512).lanes_for(&HirType::F32),
            Some(16)
        );
        assert_eq!(
            VectorProfile::of_bits(512).lanes_for(&HirType::F64),
            Some(8)
        );
    }

    /// Anything without a lane size of its own, such as a pointer or an
    /// aggregate, is not vectorized by lane count.
    #[test]
    fn a_type_without_a_lane_size_has_no_lane_count() {
        assert_eq!(
            VectorProfile::portable().lanes_for(&HirType::Ptr(Box::new(HirType::F32))),
            None
        );
        assert_eq!(VectorProfile::portable().lanes_for(&HirType::Void), None);
    }

    /// A width that is not a power of two rounds down, and nothing goes
    /// below one register.
    #[test]
    fn a_width_is_a_power_of_two_and_never_below_one_register() {
        assert_eq!(VectorProfile::of_bits(300).bits(), 256);
        assert_eq!(VectorProfile::of_bits(64).bits(), 128);
        assert_eq!(VectorProfile::of_bits(0).bits(), 128);
    }

    /// The default must remain the width every backend accepts, since a
    /// module is compiled by more than one.
    #[test]
    fn the_default_is_what_every_backend_accepts() {
        assert_eq!(VectorProfile::default().bits(), PORTABLE_BITS);
        assert_eq!(VectorProfile::portable().bits(), 128);
    }
}
