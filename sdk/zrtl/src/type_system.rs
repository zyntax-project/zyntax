//! Type System for ZRTL
//!
//! This module provides type tags, categories, and flags for identifying
//! and working with dynamically-typed values in ZRTL plugins.
//!
//! The type system is designed to be C ABI compatible for interop with
//! native code and the Zyntax JIT compiler.

/// Type categories for DynamicBox values
///
/// These match the categories defined in `zrtl.h` for C interop.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeCategory {
    Void = 0x00,
    Bool = 0x01,
    Int = 0x02,
    UInt = 0x03,
    Float = 0x04,
    String = 0x05,
    Array = 0x06,
    Map = 0x07,
    Struct = 0x08,
    Class = 0x09,
    Enum = 0x0A,
    Union = 0x0B,
    Function = 0x0C,
    Pointer = 0x0D,
    Optional = 0x0E,
    Result = 0x0F,
    Tuple = 0x10,
    TraitObject = 0x11,
    Opaque = 0x12,
    Custom = 0xFF,
}

impl TypeCategory {
    /// Convert from raw u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::Void),
            0x01 => Some(Self::Bool),
            0x02 => Some(Self::Int),
            0x03 => Some(Self::UInt),
            0x04 => Some(Self::Float),
            0x05 => Some(Self::String),
            0x06 => Some(Self::Array),
            0x07 => Some(Self::Map),
            0x08 => Some(Self::Struct),
            0x09 => Some(Self::Class),
            0x0A => Some(Self::Enum),
            0x0B => Some(Self::Union),
            0x0C => Some(Self::Function),
            0x0D => Some(Self::Pointer),
            0x0E => Some(Self::Optional),
            0x0F => Some(Self::Result),
            0x10 => Some(Self::Tuple),
            0x11 => Some(Self::TraitObject),
            0x12 => Some(Self::Opaque),
            0xFF => Some(Self::Custom),
            _ => None,
        }
    }

    /// Check if this is a primitive type
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Self::Void | Self::Bool | Self::Int | Self::UInt | Self::Float
        )
    }

    /// Check if this is a numeric type
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Int | Self::UInt | Self::Float)
    }

    /// Check if this is a collection type
    pub fn is_collection(&self) -> bool {
        matches!(self, Self::Array | Self::Map | Self::Tuple)
    }
}

/// Type flags for additional type information
///
/// Flags can be combined using bitwise OR.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TypeFlags(pub u8);

impl TypeFlags {
    pub const NONE: Self = Self(0x00);
    pub const NULLABLE: Self = Self(0x01);
    pub const MUTABLE: Self = Self(0x02);
    pub const BOXED: Self = Self(0x04);
    pub const ARC: Self = Self(0x08);
    pub const WEAK: Self = Self(0x10);
    pub const PINNED: Self = Self(0x20);
    /// Type implements Display trait (has to_string function)
    pub const DISPLAY: Self = Self(0x40);

    /// Check if a flag is set
    #[inline]
    pub fn contains(&self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Add a flag
    #[inline]
    pub fn with(&self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }

    /// Remove a flag
    #[inline]
    pub fn without(&self, flag: Self) -> Self {
        Self(self.0 & !flag.0)
    }

    /// Check if nullable
    #[inline]
    pub fn is_nullable(&self) -> bool {
        self.contains(Self::NULLABLE)
    }

    /// Check if mutable
    #[inline]
    pub fn is_mutable(&self) -> bool {
        self.contains(Self::MUTABLE)
    }

    /// Check if boxed (heap-allocated)
    #[inline]
    pub fn is_boxed(&self) -> bool {
        self.contains(Self::BOXED)
    }

    /// Check if reference-counted
    #[inline]
    pub fn is_arc(&self) -> bool {
        self.contains(Self::ARC)
    }

    /// Check if type implements Display (has to_string function)
    #[inline]
    pub fn implements_display(&self) -> bool {
        self.contains(Self::DISPLAY)
    }
}

impl std::ops::BitOr for TypeFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for TypeFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

/// Primitive size identifiers
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveSize {
    Bits8 = 0x01,
    Bits16 = 0x02,
    Bits32 = 0x03,
    Bits64 = 0x04,
    Pointer = 0x05,
}

impl PrimitiveSize {
    /// Get the size in bytes
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Bits8 => 1,
            Self::Bits16 => 2,
            Self::Bits32 => 4,
            Self::Bits64 => 8,
            Self::Pointer => std::mem::size_of::<usize>(),
        }
    }
}

/// Type tag - 32-bit packed type identifier
///
/// Layout: `[flags:8][type_id:16][category:8]`
///
/// This compact representation allows efficient type checking at runtime.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TypeTag(pub u32);

impl TypeTag {
    /// Create a new type tag
    #[inline]
    pub const fn new(category: TypeCategory, type_id: u16, flags: TypeFlags) -> Self {
        Self(((flags.0 as u32) << 24) | ((type_id as u32) << 8) | (category as u32))
    }

    /// Create a type tag from raw value
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Get the raw value
    #[inline]
    pub const fn raw(&self) -> u32 {
        self.0
    }

    /// Get the category
    #[inline]
    pub fn category(&self) -> TypeCategory {
        // SAFETY: We mask to valid range, and TypeCategory covers all valid values
        unsafe { std::mem::transmute((self.0 & 0xFF) as u8) }
    }

    /// Get the type ID within the category
    #[inline]
    pub fn type_id(&self) -> u16 {
        ((self.0 >> 8) & 0xFFFF) as u16
    }

    /// Get the flags
    #[inline]
    pub fn flags(&self) -> TypeFlags {
        TypeFlags(((self.0 >> 24) & 0xFF) as u8)
    }

    /// Check if this matches a category
    #[inline]
    pub fn is_category(&self, category: TypeCategory) -> bool {
        self.category() == category
    }

    /// Create a nullable version of this tag
    #[inline]
    pub fn nullable(self) -> Self {
        Self::new(
            self.category(),
            self.type_id(),
            self.flags().with(TypeFlags::NULLABLE),
        )
    }

    /// Create a version with Display trait
    #[inline]
    pub fn with_display(self) -> Self {
        Self::new(
            self.category(),
            self.type_id(),
            self.flags().with(TypeFlags::DISPLAY),
        )
    }

    /// Check if type implements Display
    #[inline]
    pub fn implements_display(&self) -> bool {
        self.flags().implements_display()
    }

    // Pre-defined type tags for primitives
    pub const VOID: Self = Self::new(TypeCategory::Void, 0, TypeFlags::NONE);
    pub const BOOL: Self = Self::new(TypeCategory::Bool, 0, TypeFlags::NONE);
    pub const I8: Self = Self::new(
        TypeCategory::Int,
        PrimitiveSize::Bits8 as u16,
        TypeFlags::NONE,
    );
    pub const I16: Self = Self::new(
        TypeCategory::Int,
        PrimitiveSize::Bits16 as u16,
        TypeFlags::NONE,
    );
    pub const I32: Self = Self::new(
        TypeCategory::Int,
        PrimitiveSize::Bits32 as u16,
        TypeFlags::NONE,
    );
    pub const I64: Self = Self::new(
        TypeCategory::Int,
        PrimitiveSize::Bits64 as u16,
        TypeFlags::NONE,
    );
    pub const ISIZE: Self = Self::new(
        TypeCategory::Int,
        PrimitiveSize::Pointer as u16,
        TypeFlags::NONE,
    );
    pub const U8: Self = Self::new(
        TypeCategory::UInt,
        PrimitiveSize::Bits8 as u16,
        TypeFlags::NONE,
    );
    pub const U16: Self = Self::new(
        TypeCategory::UInt,
        PrimitiveSize::Bits16 as u16,
        TypeFlags::NONE,
    );
    pub const U32: Self = Self::new(
        TypeCategory::UInt,
        PrimitiveSize::Bits32 as u16,
        TypeFlags::NONE,
    );
    pub const U64: Self = Self::new(
        TypeCategory::UInt,
        PrimitiveSize::Bits64 as u16,
        TypeFlags::NONE,
    );
    pub const USIZE: Self = Self::new(
        TypeCategory::UInt,
        PrimitiveSize::Pointer as u16,
        TypeFlags::NONE,
    );
    pub const F32: Self = Self::new(
        TypeCategory::Float,
        PrimitiveSize::Bits32 as u16,
        TypeFlags::NONE,
    );
    pub const F64: Self = Self::new(
        TypeCategory::Float,
        PrimitiveSize::Bits64 as u16,
        TypeFlags::NONE,
    );
    pub const STRING: Self = Self::new(TypeCategory::String, 0, TypeFlags::NONE);
}

/// Macro to create a type tag at compile time
#[macro_export]
macro_rules! zrtl_tag {
    (void) => {
        $crate::TypeTag::VOID
    };
    (bool) => {
        $crate::TypeTag::BOOL
    };
    (i8) => {
        $crate::TypeTag::I8
    };
    (i16) => {
        $crate::TypeTag::I16
    };
    (i32) => {
        $crate::TypeTag::I32
    };
    (i64) => {
        $crate::TypeTag::I64
    };
    (isize) => {
        $crate::TypeTag::ISIZE
    };
    (u8) => {
        $crate::TypeTag::U8
    };
    (u16) => {
        $crate::TypeTag::U16
    };
    (u32) => {
        $crate::TypeTag::U32
    };
    (u64) => {
        $crate::TypeTag::U64
    };
    (usize) => {
        $crate::TypeTag::USIZE
    };
    (f32) => {
        $crate::TypeTag::F32
    };
    (f64) => {
        $crate::TypeTag::F64
    };
    (string) => {
        $crate::TypeTag::STRING
    };
    ($category:ident, $type_id:expr) => {
        $crate::TypeTag::new(
            $crate::TypeCategory::$category,
            $type_id,
            $crate::TypeFlags::NONE,
        )
    };
    ($category:ident, $type_id:expr, $flags:expr) => {
        $crate::TypeTag::new($crate::TypeCategory::$category, $type_id, $flags)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_tag_components() {
        let tag = TypeTag::new(TypeCategory::Int, 3, TypeFlags::NULLABLE);

        assert_eq!(tag.category(), TypeCategory::Int);
        assert_eq!(tag.type_id(), 3);
        assert!(tag.flags().is_nullable());
    }

    #[test]
    fn test_predefined_tags() {
        assert_eq!(TypeTag::I32.category(), TypeCategory::Int);
        assert_eq!(TypeTag::I32.type_id(), PrimitiveSize::Bits32 as u16);

        assert_eq!(TypeTag::F64.category(), TypeCategory::Float);
        assert_eq!(TypeTag::STRING.category(), TypeCategory::String);
    }

    #[test]
    fn test_type_flags() {
        let flags = TypeFlags::NULLABLE | TypeFlags::MUTABLE;
        assert!(flags.is_nullable());
        assert!(flags.is_mutable());
        assert!(!flags.is_boxed());

        let without_nullable = flags.without(TypeFlags::NULLABLE);
        assert!(!without_nullable.is_nullable());
        assert!(without_nullable.is_mutable());
    }

    #[test]
    fn test_display_flag() {
        let flags = TypeFlags::DISPLAY;
        assert!(flags.implements_display());
        assert!(!TypeFlags::NONE.implements_display());

        // Test with_display on TypeTag
        let opaque_tag = TypeTag::new(TypeCategory::Opaque, 1, TypeFlags::NONE);
        assert!(!opaque_tag.implements_display());

        let displayable = opaque_tag.with_display();
        assert!(displayable.implements_display());
        assert_eq!(displayable.category(), TypeCategory::Opaque);
        assert_eq!(displayable.type_id(), 1);
    }

    #[test]
    fn test_zrtl_tag_macro() {
        assert_eq!(zrtl_tag!(i32), TypeTag::I32);
        assert_eq!(zrtl_tag!(string), TypeTag::STRING);

        let custom = zrtl_tag!(Struct, 42);
        assert_eq!(custom.category(), TypeCategory::Struct);
        assert_eq!(custom.type_id(), 42);
    }
}
