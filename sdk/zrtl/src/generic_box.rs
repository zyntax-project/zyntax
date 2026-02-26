//! GenericBox - Type-parameterized boxed values
//!
//! For generic types like `Array<T>`, `Map<K,V>`, `Optional<T>`, etc.,
//! we need to track the type arguments alongside the value.

use crate::dynamic_box::{DropFn, DynamicBox};
use crate::type_system::{TypeCategory, TypeFlags, TypeTag};

/// Maximum type arguments for a generic type
pub const MAX_TYPE_ARGS: usize = 8;

/// Generic type arguments
///
/// Tracks the type parameters for generic types like `Array<Int32>` or `Map<String, Array<Int32>>`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GenericTypeArgs {
    /// Number of type arguments
    pub count: u8,
    /// Type tags for each argument
    pub args: [TypeTag; MAX_TYPE_ARGS],
    /// Nested generic args (for nested generics like `Array<Map<K,V>>`)
    pub nested: [Option<Box<GenericTypeArgs>>; MAX_TYPE_ARGS],
}

impl GenericTypeArgs {
    /// Create empty type args
    pub fn new(count: u8) -> Self {
        Self {
            count,
            args: [TypeTag::VOID; MAX_TYPE_ARGS],
            nested: Default::default(),
        }
    }

    /// Create type args for `Array<T>`
    pub fn array(element_type: TypeTag) -> Self {
        let mut args = Self::new(1);
        args.args[0] = element_type;
        args
    }

    /// Create type args for `Array<T>` with nested generic element
    pub fn array_generic(element_args: GenericTypeArgs) -> Self {
        let mut args = Self::new(1);
        args.args[0] = TypeTag::new(TypeCategory::Array, 0, TypeFlags::NONE);
        args.nested[0] = Some(Box::new(element_args));
        args
    }

    /// Create type args for `Map<K, V>`
    pub fn map(key_type: TypeTag, value_type: TypeTag) -> Self {
        let mut args = Self::new(2);
        args.args[0] = key_type;
        args.args[1] = value_type;
        args
    }

    /// Create type args for `Optional<T>`
    pub fn optional(inner_type: TypeTag) -> Self {
        let mut args = Self::new(1);
        args.args[0] = inner_type;
        args
    }

    /// Create type args for `Result<T, E>`
    pub fn result(ok_type: TypeTag, err_type: TypeTag) -> Self {
        let mut args = Self::new(2);
        args.args[0] = ok_type;
        args.args[1] = err_type;
        args
    }

    /// Get type argument at index
    pub fn get(&self, index: usize) -> Option<TypeTag> {
        if index < self.count as usize {
            Some(self.args[index])
        } else {
            None
        }
    }

    /// Get nested generic args at index
    pub fn get_nested(&self, index: usize) -> Option<&GenericTypeArgs> {
        if index < self.count as usize {
            self.nested[index].as_ref().map(|b| b.as_ref())
        } else {
            None
        }
    }

    /// Check if any type argument has nested generics
    pub fn has_nested(&self) -> bool {
        self.nested.iter().any(|n| n.is_some())
    }
}

impl Default for GenericTypeArgs {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Extended DynamicBox with generic type information
///
/// Use this for generic container types like arrays and maps.
#[repr(C)]
#[derive(Debug)]
pub struct GenericBox {
    /// Base dynamic box
    pub base: DynamicBox,
    /// Generic type arguments (null if non-generic)
    pub type_args: Option<Box<GenericTypeArgs>>,
}

impl GenericBox {
    /// Create a null generic box
    pub fn null() -> Self {
        Self {
            base: DynamicBox::null(),
            type_args: None,
        }
    }

    /// Create from a DynamicBox without generics
    pub fn from_base(base: DynamicBox) -> Self {
        Self {
            base,
            type_args: None,
        }
    }

    /// Create from a DynamicBox with generic type args
    pub fn with_type_args(base: DynamicBox, args: GenericTypeArgs) -> Self {
        Self {
            base,
            type_args: Some(Box::new(args)),
        }
    }

    /// Check if this is a generic type
    pub fn is_generic(&self) -> bool {
        self.type_args
            .as_ref()
            .map(|a| a.count > 0)
            .unwrap_or(false)
    }

    /// Get the type arguments
    pub fn type_args(&self) -> Option<&GenericTypeArgs> {
        self.type_args.as_ref().map(|b| b.as_ref())
    }

    /// Get type argument at index
    pub fn type_arg(&self, index: usize) -> Option<TypeTag> {
        self.type_args.as_ref().and_then(|args| args.get(index))
    }

    /// Create a generic box for `Array<T>`
    pub fn array(data: *mut u8, size: u32, element_type: TypeTag, dropper: Option<DropFn>) -> Self {
        let base = DynamicBox {
            tag: TypeTag::new(TypeCategory::Array, 0, TypeFlags::NONE),
            size,
            data,
            dropper,
            display_fn: None,
        };
        Self::with_type_args(base, GenericTypeArgs::array(element_type))
    }

    /// Create a generic box for `Map<K, V>`
    pub fn map(
        data: *mut u8,
        size: u32,
        key_type: TypeTag,
        value_type: TypeTag,
        dropper: Option<DropFn>,
    ) -> Self {
        let base = DynamicBox {
            tag: TypeTag::new(TypeCategory::Map, 0, TypeFlags::NONE),
            size,
            data,
            dropper,
            display_fn: None,
        };
        Self::with_type_args(base, GenericTypeArgs::map(key_type, value_type))
    }

    /// Create a generic box for `Optional<T>`
    pub fn optional(
        data: *mut u8,
        size: u32,
        inner_type: TypeTag,
        dropper: Option<DropFn>,
    ) -> Self {
        let base = DynamicBox {
            tag: TypeTag::new(TypeCategory::Optional, 0, TypeFlags::NONE),
            size,
            data,
            dropper,
            display_fn: None,
        };
        Self::with_type_args(base, GenericTypeArgs::optional(inner_type))
    }

    /// Create a generic box for `Result<T, E>`
    pub fn result(
        data: *mut u8,
        size: u32,
        ok_type: TypeTag,
        err_type: TypeTag,
        dropper: Option<DropFn>,
    ) -> Self {
        let base = DynamicBox {
            tag: TypeTag::new(TypeCategory::Result, 0, TypeFlags::NONE),
            size,
            data,
            dropper,
            display_fn: None,
        };
        Self::with_type_args(base, GenericTypeArgs::result(ok_type, err_type))
    }

    /// Free the generic box
    pub fn free(&mut self) {
        self.base.free();
        self.type_args = None;
    }
}

impl Default for GenericBox {
    fn default() -> Self {
        Self::null()
    }
}

impl Clone for GenericBox {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone_box(),
            type_args: self.type_args.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_type_args() {
        let args = GenericTypeArgs::array(TypeTag::I32);
        assert_eq!(args.count, 1);
        assert_eq!(args.get(0), Some(TypeTag::I32));
        assert_eq!(args.get(1), None);
    }

    #[test]
    fn test_map_type_args() {
        let args = GenericTypeArgs::map(TypeTag::STRING, TypeTag::I64);
        assert_eq!(args.count, 2);
        assert_eq!(args.get(0), Some(TypeTag::STRING));
        assert_eq!(args.get(1), Some(TypeTag::I64));
    }

    #[test]
    fn test_nested_generic() {
        // Array<Array<Int32>>
        let inner = GenericTypeArgs::array(TypeTag::I32);
        let outer = GenericTypeArgs::array_generic(inner);

        assert!(outer.has_nested());
        let nested = outer.get_nested(0).unwrap();
        assert_eq!(nested.get(0), Some(TypeTag::I32));
    }

    #[test]
    fn test_generic_box() {
        let mut value: i32 = 42;
        let base = DynamicBox::box_i32(&mut value);
        let gbox = GenericBox::from_base(base);

        assert!(!gbox.is_generic());
    }
}
