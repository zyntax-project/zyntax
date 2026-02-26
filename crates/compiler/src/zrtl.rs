//! ZRTL (Zyntax Runtime Library) - Dynamic plugin format
//!
//! Similar to HashLink's HDLL format, ZRTL files are native dynamic libraries
//! (.so/.dylib/.dll) that export runtime symbols for the JIT/AOT compiler.
//!
//! # Creating a ZRTL Plugin
//!
//! ```c
//! // my_runtime.c
//! #include "zrtl.h"
//!
//! int32_t my_add(int32_t a, int32_t b) {
//!     return a + b;
//! }
//!
//! void my_print(const char* msg) {
//!     printf("%s\n", msg);
//! }
//!
//! // Symbol table - must be named _zrtl_symbols
//! ZRTL_EXPORT ZrtlSymbol _zrtl_symbols[] = {
//!     { "$MyRuntime$add", (void*)my_add },
//!     { "$MyRuntime$print", (void*)my_print },
//!     { NULL, NULL }  // Sentinel
//! };
//!
//! // Plugin info - must be named _zrtl_info
//! ZRTL_EXPORT ZrtlInfo _zrtl_info = {
//!     .version = ZRTL_VERSION,
//!     .name = "my_runtime",
//! };
//! ```
//!
//! Compile as shared library:
//! ```bash
//! # Linux
//! gcc -shared -fPIC -o my_runtime.zrtl my_runtime.c
//!
//! # macOS
//! clang -shared -fPIC -o my_runtime.zrtl my_runtime.c
//!
//! # Windows
//! cl /LD my_runtime.c /Fe:my_runtime.zrtl
//! ```
//!
//! # Loading ZRTL Plugins
//!
//! ```bash
//! zyntax compile --runtime my_runtime.zrtl --source input.hx
//! ```

use std::ffi::{CStr, OsStr};
use std::path::Path;

/// Current ZRTL format version
pub const ZRTL_VERSION: u32 = 1;

// ============================================================
// Type Tags for Dynamic Type Identification
// ============================================================

/// Type categories for DynamicBox
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

/// Type flags
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

    pub fn contains(&self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    pub fn with(&self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }
}

/// Primitive size IDs
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveSize {
    Bits8 = 0x01,
    Bits16 = 0x02,
    Bits32 = 0x03,
    Bits64 = 0x04,
    Pointer = 0x05,
}

/// Type tag - 32-bit packed type identifier
/// Layout: [flags:8][type_id:16][category:8]
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TypeTag(pub u32);

impl TypeTag {
    /// Create a new type tag
    pub const fn new(category: TypeCategory, type_id: u16, flags: TypeFlags) -> Self {
        Self(((flags.0 as u32) << 24) | ((type_id as u32) << 8) | (category as u32))
    }

    /// Get the category
    pub fn category(&self) -> TypeCategory {
        unsafe { std::mem::transmute((self.0 & 0xFF) as u8) }
    }

    /// Get the type ID
    pub fn type_id(&self) -> u16 {
        ((self.0 >> 8) & 0xFFFF) as u16
    }

    /// Get the flags
    pub fn flags(&self) -> TypeFlags {
        TypeFlags(((self.0 >> 24) & 0xFF) as u8)
    }

    /// Check if this matches a category
    pub fn is_category(&self, category: TypeCategory) -> bool {
        self.category() == category
    }

    // Pre-defined type tags
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

// ============================================================
// DynamicValue - Runtime Tagged Pointer (C ABI compatible)
// ============================================================

/// Type ID for runtime type identification
///
/// This is a unique identifier for each type in the runtime.
/// Built-in types have well-known IDs, custom types get dynamically assigned IDs.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TypeId(pub u32);

impl TypeId {
    /// Create a new type ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Create a type ID from category and sub-id
    pub const fn from_parts(category: TypeCategory, sub_id: u16) -> Self {
        Self(((category as u32) << 16) | (sub_id as u32))
    }

    /// Get the category
    pub fn category(&self) -> TypeCategory {
        unsafe { std::mem::transmute(((self.0 >> 16) & 0xFF) as u8) }
    }

    /// Get the sub-id within the category
    pub fn sub_id(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    // Built-in type IDs
    pub const VOID: Self = Self::from_parts(TypeCategory::Void, 0);
    pub const NULL: Self = Self::from_parts(TypeCategory::Void, 1);
    pub const BOOL: Self = Self::from_parts(TypeCategory::Bool, 0);
    pub const I8: Self = Self::from_parts(TypeCategory::Int, PrimitiveSize::Bits8 as u16);
    pub const I16: Self = Self::from_parts(TypeCategory::Int, PrimitiveSize::Bits16 as u16);
    pub const I32: Self = Self::from_parts(TypeCategory::Int, PrimitiveSize::Bits32 as u16);
    pub const I64: Self = Self::from_parts(TypeCategory::Int, PrimitiveSize::Bits64 as u16);
    pub const ISIZE: Self = Self::from_parts(TypeCategory::Int, PrimitiveSize::Pointer as u16);
    pub const U8: Self = Self::from_parts(TypeCategory::UInt, PrimitiveSize::Bits8 as u16);
    pub const U16: Self = Self::from_parts(TypeCategory::UInt, PrimitiveSize::Bits16 as u16);
    pub const U32: Self = Self::from_parts(TypeCategory::UInt, PrimitiveSize::Bits32 as u16);
    pub const U64: Self = Self::from_parts(TypeCategory::UInt, PrimitiveSize::Bits64 as u16);
    pub const USIZE: Self = Self::from_parts(TypeCategory::UInt, PrimitiveSize::Pointer as u16);
    pub const F32: Self = Self::from_parts(TypeCategory::Float, PrimitiveSize::Bits32 as u16);
    pub const F64: Self = Self::from_parts(TypeCategory::Float, PrimitiveSize::Bits64 as u16);
    pub const STRING: Self = Self::from_parts(TypeCategory::String, 0);
    pub const DYNAMIC: Self = Self::from_parts(TypeCategory::Opaque, 0);
}

/// Drop function type for custom types (C ABI)
pub type DropFn = extern "C" fn(*mut u8);

/// Type metadata stored with each DynamicValue
///
/// This provides full runtime type information including:
/// - Type ID for quick equality checks
/// - Type category for type-safe operations
/// - Generic type arguments for parameterized types
/// - Drop function for cleanup
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TypeMeta {
    /// Type ID for quick identification
    pub type_id: TypeId,
    /// Size of the value in bytes
    pub size: u32,
    /// Drop function (null if no cleanup needed)
    pub dropper: Option<DropFn>,
    /// Pointer to generic type args (null if non-generic)
    pub generic_args: *const GenericTypeArgs,
}

impl TypeMeta {
    /// Create a null type metadata
    pub const fn null() -> Self {
        Self {
            type_id: TypeId::NULL,
            size: 0,
            dropper: None,
            generic_args: std::ptr::null(),
        }
    }

    /// Create type metadata for a primitive type
    pub const fn primitive(type_id: TypeId, size: u32) -> Self {
        Self {
            type_id,
            size,
            dropper: None,
            generic_args: std::ptr::null(),
        }
    }

    /// Create type metadata with a drop function
    pub const fn with_dropper(type_id: TypeId, size: u32, dropper: DropFn) -> Self {
        Self {
            type_id,
            size,
            dropper: Some(dropper),
            generic_args: std::ptr::null(),
        }
    }

    // Pre-defined type metadata for primitives
    pub const META_VOID: Self = Self::null();
    pub const META_BOOL: Self = Self::primitive(TypeId::BOOL, 1);
    pub const META_I8: Self = Self::primitive(TypeId::I8, 1);
    pub const META_I16: Self = Self::primitive(TypeId::I16, 2);
    pub const META_I32: Self = Self::primitive(TypeId::I32, 4);
    pub const META_I64: Self = Self::primitive(TypeId::I64, 8);
    pub const META_U8: Self = Self::primitive(TypeId::U8, 1);
    pub const META_U16: Self = Self::primitive(TypeId::U16, 2);
    pub const META_U32: Self = Self::primitive(TypeId::U32, 4);
    pub const META_U64: Self = Self::primitive(TypeId::U64, 8);
    pub const META_F32: Self = Self::primitive(TypeId::F32, 4);
    pub const META_F64: Self = Self::primitive(TypeId::F64, 8);
}

impl Default for TypeMeta {
    fn default() -> Self {
        Self::null()
    }
}

// TypeMeta is safe to share across threads because:
// - The dropper function pointer is thread-safe (extern "C" functions are)
// - The generic_args pointer is only read, never mutated through TypeMeta
// - All other fields are Copy types
unsafe impl Sync for TypeMeta {}
unsafe impl Send for TypeMeta {}

/// Dynamic value: tagged union of (type_meta_ptr, value_ptr)
///
/// This is the runtime representation of Haxe's Dynamic type.
/// The value_ptr points to heap-allocated memory containing the actual value.
/// The type_meta_ptr points to type metadata (can be static or heap-allocated).
///
/// Layout (16 bytes on 64-bit):
/// - type_meta_ptr: 8 bytes pointer to TypeMeta
/// - value_ptr: 8 bytes pointer to the actual data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DynamicValue {
    /// Pointer to type metadata (usually points to static TypeMeta)
    pub type_meta: *const TypeMeta,
    /// Pointer to the heap-allocated value (null for void/null types)
    pub value_ptr: *mut u8,
}

// Static type metadata for built-in types
static META_NULL: TypeMeta = TypeMeta::null();
static META_VOID: TypeMeta = TypeMeta::primitive(TypeId::VOID, 0);
static META_BOOL: TypeMeta = TypeMeta::primitive(TypeId::BOOL, 1);
static META_I32: TypeMeta = TypeMeta::primitive(TypeId::I32, 4);
static META_I64: TypeMeta = TypeMeta::primitive(TypeId::I64, 8);
static META_F32: TypeMeta = TypeMeta::primitive(TypeId::F32, 4);
static META_F64: TypeMeta = TypeMeta::primitive(TypeId::F64, 8);
static META_STRING: TypeMeta =
    TypeMeta::primitive(TypeId::STRING, std::mem::size_of::<String>() as u32);

impl DynamicValue {
    /// Create a null dynamic value
    pub const fn null() -> Self {
        Self {
            type_meta: &META_NULL as *const TypeMeta,
            value_ptr: std::ptr::null_mut(),
        }
    }

    /// Create a void dynamic value
    pub const fn void() -> Self {
        Self {
            type_meta: &META_VOID as *const TypeMeta,
            value_ptr: std::ptr::null_mut(),
        }
    }

    /// Check if this is a null value
    pub fn is_null(&self) -> bool {
        self.value_ptr.is_null()
    }

    /// Get the type metadata
    ///
    /// # Safety
    /// The type_meta pointer must be valid
    pub unsafe fn meta(&self) -> &TypeMeta {
        &*self.type_meta
    }

    /// Get the type ID
    pub fn type_id(&self) -> TypeId {
        unsafe { (*self.type_meta).type_id }
    }

    /// Check if this matches a type category
    pub fn is_category(&self, category: TypeCategory) -> bool {
        self.type_id().category() == category
    }

    /// Check if this matches a specific type
    pub fn is_type(&self, type_id: TypeId) -> bool {
        self.type_id() == type_id
    }

    /// Get the generic type arguments if present
    ///
    /// # Safety
    /// The type_meta and generic_args pointers must be valid
    pub unsafe fn generic_args(&self) -> Option<&GenericTypeArgs> {
        let meta = self.meta();
        if meta.generic_args.is_null() {
            None
        } else {
            Some(&*meta.generic_args)
        }
    }

    /// Create a dynamic value from a boxed value with type metadata
    pub fn from_box_with_meta<T>(value: Box<T>, meta: &'static TypeMeta) -> Self {
        Self {
            type_meta: meta,
            value_ptr: Box::into_raw(value) as *mut u8,
        }
    }

    /// Create a dynamic value from a boxed value with custom type metadata pointer
    ///
    /// # Safety
    /// The meta pointer must remain valid for the lifetime of this DynamicValue
    pub unsafe fn from_box_with_meta_ptr<T>(value: Box<T>, meta: *const TypeMeta) -> Self {
        Self {
            type_meta: meta,
            value_ptr: Box::into_raw(value) as *mut u8,
        }
    }

    /// Create a dynamic value for a primitive type
    pub fn from_i32(value: i32) -> Self {
        Self::from_box_with_meta(Box::new(value), &META_I32)
    }

    pub fn from_i64(value: i64) -> Self {
        Self::from_box_with_meta(Box::new(value), &META_I64)
    }

    pub fn from_f32(value: f32) -> Self {
        Self::from_box_with_meta(Box::new(value), &META_F32)
    }

    pub fn from_f64(value: f64) -> Self {
        Self::from_box_with_meta(Box::new(value), &META_F64)
    }

    pub fn from_bool(value: bool) -> Self {
        Self::from_box_with_meta(Box::new(value), &META_BOOL)
    }

    pub fn from_string(value: String) -> Self {
        Self::from_box_with_meta(Box::new(value), &META_STRING)
    }

    /// Get the value as a typed reference
    ///
    /// # Safety
    /// Caller must ensure type matches T
    pub unsafe fn as_ref<T>(&self) -> Option<&T> {
        if self.value_ptr.is_null() {
            None
        } else {
            Some(&*(self.value_ptr as *const T))
        }
    }

    /// Get the value as a typed mutable reference
    ///
    /// # Safety
    /// Caller must ensure type matches T
    pub unsafe fn as_mut<T>(&mut self) -> Option<&mut T> {
        if self.value_ptr.is_null() {
            None
        } else {
            Some(&mut *(self.value_ptr as *mut T))
        }
    }

    /// Take ownership of the boxed value
    ///
    /// # Safety
    /// Caller must ensure type matches T, and must not use this DynamicValue again
    pub unsafe fn into_box<T>(self) -> Option<Box<T>> {
        if self.value_ptr.is_null() {
            None
        } else {
            Some(Box::from_raw(self.value_ptr as *mut T))
        }
    }

    /// Drop the contained value using the type's drop function
    ///
    /// # Safety
    /// This should be called when the DynamicValue is no longer needed.
    pub unsafe fn drop_value(&mut self) {
        if !self.value_ptr.is_null() {
            let meta = self.meta();
            if let Some(dropper) = meta.dropper {
                dropper(self.value_ptr);
            }
            self.value_ptr = std::ptr::null_mut();
            self.type_meta = &META_NULL;
        }
    }

    /// Drop the contained value assuming it's type T
    ///
    /// # Safety
    /// Caller must ensure the actual type is T
    pub unsafe fn drop_as<T>(&mut self) {
        if !self.value_ptr.is_null() {
            let _ = Box::from_raw(self.value_ptr as *mut T);
            self.value_ptr = std::ptr::null_mut();
            self.type_meta = &META_NULL;
        }
    }

    /// Safe accessors for primitive types (with type checking)
    pub fn get_i32(&self) -> Option<i32> {
        if self.type_id() == TypeId::I32 {
            unsafe { self.as_ref::<i32>().copied() }
        } else {
            None
        }
    }

    pub fn get_i64(&self) -> Option<i64> {
        if self.type_id() == TypeId::I64 {
            unsafe { self.as_ref::<i64>().copied() }
        } else {
            None
        }
    }

    pub fn get_f32(&self) -> Option<f32> {
        if self.type_id() == TypeId::F32 {
            unsafe { self.as_ref::<f32>().copied() }
        } else {
            None
        }
    }

    pub fn get_f64(&self) -> Option<f64> {
        if self.type_id() == TypeId::F64 {
            unsafe { self.as_ref::<f64>().copied() }
        } else {
            None
        }
    }

    pub fn get_bool(&self) -> Option<bool> {
        if self.type_id() == TypeId::BOOL {
            unsafe { self.as_ref::<bool>().copied() }
        } else {
            None
        }
    }

    pub fn get_string(&self) -> Option<&String> {
        if self.type_id() == TypeId::STRING {
            unsafe { self.as_ref::<String>() }
        } else {
            None
        }
    }
}

impl Default for DynamicValue {
    fn default() -> Self {
        Self::null()
    }
}

// Note: DynamicValue does NOT implement Drop automatically!
// The user must explicitly call drop_value() or drop_as<T>() to free memory.
// This is intentional for C ABI compatibility and ownership transfer.

// ============================================================
// Generic Type Arguments
// ============================================================

/// Maximum type arguments for a generic type
pub const MAX_TYPE_ARGS: usize = 8;

/// Generic type arguments for parameterized types
#[derive(Debug, Clone)]
pub struct GenericTypeArgs {
    /// Number of type arguments
    pub count: u8,
    /// Type IDs for each argument
    pub args: [TypeId; MAX_TYPE_ARGS],
    /// Nested generic args (for nested generics like Array<Map<K,V>>)
    pub nested: [Option<Box<GenericTypeArgs>>; MAX_TYPE_ARGS],
}

impl Default for GenericTypeArgs {
    fn default() -> Self {
        Self {
            count: 0,
            args: [TypeId::VOID; MAX_TYPE_ARGS],
            nested: Default::default(),
        }
    }
}

impl GenericTypeArgs {
    /// Create new generic type args with given count
    pub fn new(count: u8) -> Self {
        Self {
            count,
            ..Default::default()
        }
    }

    /// Create for Array<T>
    pub fn array(element_type: TypeId) -> Self {
        let mut args = Self::new(1);
        args.args[0] = element_type;
        args
    }

    /// Create for Map<K, V>
    pub fn map(key_type: TypeId, value_type: TypeId) -> Self {
        let mut args = Self::new(2);
        args.args[0] = key_type;
        args.args[1] = value_type;
        args
    }

    /// Create for Optional<T>
    pub fn optional(inner_type: TypeId) -> Self {
        let mut args = Self::new(1);
        args.args[0] = inner_type;
        args
    }

    /// Create for Result<T, E>
    pub fn result(ok_type: TypeId, err_type: TypeId) -> Self {
        let mut args = Self::new(2);
        args.args[0] = ok_type;
        args.args[1] = err_type;
        args
    }
}

/// Extended DynamicValue with generic type info
#[derive(Debug, Clone, Copy)]
pub struct GenericValue {
    /// Base dynamic value
    pub base: DynamicValue,
    /// Pointer to generic type arguments (null if non-generic)
    /// This points to heap-allocated GenericTypeArgs
    pub type_args_ptr: *const GenericTypeArgs,
}

impl GenericValue {
    /// Create a null generic value
    pub const fn null() -> Self {
        Self {
            base: DynamicValue::null(),
            type_args_ptr: std::ptr::null(),
        }
    }

    /// Create from a DynamicValue without generics
    pub const fn from_value(value: DynamicValue) -> Self {
        Self {
            base: value,
            type_args_ptr: std::ptr::null(),
        }
    }

    /// Create from a DynamicValue with generic type args
    pub fn with_type_args(value: DynamicValue, args: Box<GenericTypeArgs>) -> Self {
        Self {
            base: value,
            type_args_ptr: Box::into_raw(args),
        }
    }

    /// Check if this is a generic type
    pub fn is_generic(&self) -> bool {
        !self.type_args_ptr.is_null()
    }

    /// Get the generic type arguments
    ///
    /// # Safety
    /// The type_args_ptr must be valid if not null
    pub unsafe fn type_args(&self) -> Option<&GenericTypeArgs> {
        if self.type_args_ptr.is_null() {
            None
        } else {
            Some(&*self.type_args_ptr)
        }
    }

    /// Get type argument at index
    ///
    /// # Safety
    /// The type_args_ptr must be valid if not null
    pub unsafe fn type_arg(&self, index: usize) -> Option<TypeId> {
        self.type_args().and_then(|args| {
            if index < args.count as usize {
                Some(args.args[index])
            } else {
                None
            }
        })
    }

    /// Drop the generic type args if present
    ///
    /// # Safety
    /// Should only be called once, when the GenericValue is no longer needed
    pub unsafe fn drop_type_args(&mut self) {
        if !self.type_args_ptr.is_null() {
            let _ = Box::from_raw(self.type_args_ptr as *mut GenericTypeArgs);
            self.type_args_ptr = std::ptr::null();
        }
    }
}

impl Default for GenericValue {
    fn default() -> Self {
        Self::null()
    }
}

// ============================================================
// Type Registry for Custom Types
// ============================================================

/// Type info for registered custom types
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Size in bytes
    pub size: u32,
    /// Alignment requirement
    pub alignment: u32,
    /// Optional destructor (C ABI function pointer)
    pub dropper: Option<DropFn>,
    /// Type category
    pub category: TypeCategory,
}

/// Registry for custom runtime types
#[derive(Debug, Default)]
pub struct TypeRegistry {
    types: Vec<TypeInfo>,
    /// Base ID for custom types (to avoid collision with built-in types)
    base_id: u32,
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            // Custom types start at 0x10000 to avoid collision with built-in type IDs
            base_id: 0x00010000,
        }
    }

    /// Register a custom type, returns its TypeId
    pub fn register(&mut self, info: TypeInfo) -> TypeId {
        let id = self.base_id + self.types.len() as u32;
        self.types.push(info);
        TypeId::new(id)
    }

    /// Get type info by TypeId
    pub fn get(&self, type_id: TypeId) -> Option<&TypeInfo> {
        let id = type_id.0;
        if id >= self.base_id {
            self.types.get((id - self.base_id) as usize)
        } else {
            None
        }
    }

    /// Get the drop function for a type
    pub fn get_dropper(&self, type_id: TypeId) -> Option<DropFn> {
        self.get(type_id).and_then(|info| info.dropper)
    }

    /// Drop a DynamicValue using the registry
    ///
    /// # Safety
    /// The DynamicValue must have been created with a type_id from this registry
    pub unsafe fn drop_value(&self, value: &mut DynamicValue) {
        if !value.value_ptr.is_null() {
            if let Some(dropper) = self.get_dropper(value.type_id()) {
                dropper(value.value_ptr);
            }
            value.value_ptr = std::ptr::null_mut();
            value.type_meta = &META_NULL;
        }
    }
}

/// File extension for ZRTL files
#[cfg(target_os = "windows")]
pub const ZRTL_EXTENSION: &str = "zrtl";

#[cfg(not(target_os = "windows"))]
pub const ZRTL_EXTENSION: &str = "zrtl";

/// Maximum number of parameters in a function signature
pub const MAX_PARAMS: usize = 16;

/// Function signature for ZRTL symbols (C ABI compatible)
///
/// Describes parameter and return types for a plugin function.
/// The compiler uses this to determine whether to auto-box arguments.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ZrtlSymbolSig {
    /// Number of parameters
    pub param_count: u8,
    /// Flags for the function signature
    pub flags: ZrtlSigFlags,
    /// Return type (VOID if no return)
    pub return_type: TypeTag,
    /// Parameter types (up to MAX_PARAMS)
    pub params: [TypeTag; MAX_PARAMS],
}

impl ZrtlSymbolSig {
    /// Check if parameter at index expects DynamicBox
    pub fn param_is_dynamic(&self, index: usize) -> bool {
        if index >= self.param_count as usize {
            return false;
        }
        self.params[index].is_category(TypeCategory::Opaque)
            || self.flags.contains(ZrtlSigFlags::ALL_DYNAMIC)
    }

    /// Check if return type is DynamicBox
    pub fn returns_dynamic(&self) -> bool {
        self.return_type.is_category(TypeCategory::Opaque)
    }
}

/// Flags for function signatures
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZrtlSigFlags(pub u8);

impl ZrtlSigFlags {
    /// No flags
    pub const NONE: Self = Self(0);
    /// All parameters should be passed as DynamicBox
    pub const ALL_DYNAMIC: Self = Self(1 << 0);
    /// Function is variadic
    pub const VARIADIC: Self = Self(1 << 1);
    /// Function may have side effects
    pub const EFFECTFUL: Self = Self(1 << 2);

    /// Check if flag is set
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }
}

/// Symbol entry in the ZRTL symbol table (C ABI compatible)
#[repr(C)]
pub struct ZrtlSymbol {
    /// Symbol name (null-terminated C string)
    /// Convention: "$TypeName$method_name"
    pub name: *const std::ffi::c_char,
    /// Function pointer
    pub ptr: *const u8,
    /// Optional pointer to function signature (null if not provided)
    pub sig: *const ZrtlSymbolSig,
}

impl ZrtlSymbol {
    /// Check if this symbol has signature information
    pub fn has_sig(&self) -> bool {
        !self.sig.is_null()
    }

    /// Get the signature if available
    pub unsafe fn get_sig(&self) -> Option<&ZrtlSymbolSig> {
        if self.sig.is_null() {
            None
        } else {
            Some(&*self.sig)
        }
    }
}

// SAFETY: ZrtlSymbol contains only immutable pointers to static data:
// - name: const pointer to static C string
// - ptr: const pointer to static function
// - sig: const pointer to static signature
// These are inherently thread-safe as they cannot be modified.
unsafe impl Sync for ZrtlSymbol {}
unsafe impl Send for ZrtlSymbol {}

/// Plugin metadata (C ABI compatible)
#[repr(C)]
pub struct ZrtlInfo {
    /// ZRTL format version (must match ZRTL_VERSION)
    pub version: u32,
    /// Plugin name (null-terminated C string)
    pub name: *const std::ffi::c_char,
}

// SAFETY: ZrtlInfo contains only:
// - version: u32 (Copy, Sync)
// - name: const pointer to static C string (immutable, thread-safe)
unsafe impl Sync for ZrtlInfo {}
unsafe impl Send for ZrtlInfo {}

/// Symbol with optional signature information
#[derive(Debug, Clone)]
pub struct RuntimeSymbolInfo {
    /// Symbol name
    pub name: &'static str,
    /// Function pointer
    pub ptr: *const u8,
    /// Optional signature (for auto-boxing support)
    pub sig: Option<ZrtlSymbolSig>,
}

// SAFETY: RuntimeSymbolInfo contains only immutable pointers to static data
unsafe impl Sync for RuntimeSymbolInfo {}
unsafe impl Send for RuntimeSymbolInfo {}

/// A loaded ZRTL plugin
pub struct ZrtlPlugin {
    /// The loaded dynamic library (kept alive to prevent unloading)
    #[allow(dead_code)]
    library: libloading::Library,
    /// Plugin name
    name: String,
    /// Collected symbols with signature info
    symbols_with_sig: Vec<RuntimeSymbolInfo>,
    /// Legacy symbols list (for backwards compatibility)
    symbols: Vec<(&'static str, *const u8)>,
}

impl ZrtlPlugin {
    /// Load a ZRTL plugin from a file path
    ///
    /// # Safety
    /// The plugin library must be valid and not modified while loaded.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ZrtlError> {
        let path = path.as_ref();

        // Verify extension
        let ext = path.extension().and_then(OsStr::to_str);
        if ext != Some(ZRTL_EXTENSION)
            && ext != Some("so")
            && ext != Some("dylib")
            && ext != Some("dll")
        {
            log::warn!(
                "ZRTL plugin '{}' has unexpected extension (expected .{})",
                path.display(),
                ZRTL_EXTENSION
            );
        }

        unsafe {
            // Load the dynamic library
            let library = libloading::Library::new(path).map_err(|e| ZrtlError::LoadFailed {
                path: path.to_path_buf(),
                reason: e.to_string(),
            })?;

            // Get plugin info
            let info_sym: libloading::Symbol<*const ZrtlInfo> = library
                .get(b"_zrtl_info\0")
                .map_err(|_| ZrtlError::MissingInfo {
                    path: path.to_path_buf(),
                })?;

            let info = &**info_sym;

            // Verify version
            if info.version != ZRTL_VERSION {
                return Err(ZrtlError::VersionMismatch {
                    path: path.to_path_buf(),
                    expected: ZRTL_VERSION,
                    found: info.version,
                });
            }

            // Get plugin name
            let name = if info.name.is_null() {
                path.file_stem()
                    .and_then(OsStr::to_str)
                    .unwrap_or("unknown")
                    .to_string()
            } else {
                CStr::from_ptr(info.name)
                    .to_str()
                    .map_err(|_| ZrtlError::InvalidUtf8 {
                        path: path.to_path_buf(),
                        field: "name".to_string(),
                    })?
                    .to_string()
            };

            // Get symbol table
            let symbols_sym: libloading::Symbol<*const ZrtlSymbol> = library
                .get(b"_zrtl_symbols\0")
                .map_err(|_| ZrtlError::MissingSymbols {
                    path: path.to_path_buf(),
                })?;

            // Collect symbols until sentinel (null name)
            let mut symbols = Vec::new();
            let mut symbols_with_sig = Vec::new();
            let mut sym_ptr = *symbols_sym;

            while !(*sym_ptr).name.is_null() {
                let sym_name = CStr::from_ptr((*sym_ptr).name).to_str().map_err(|_| {
                    ZrtlError::InvalidUtf8 {
                        path: path.to_path_buf(),
                        field: "symbol name".to_string(),
                    }
                })?;

                // Leak the string to get a 'static lifetime
                // This is intentional - symbols live for the program's lifetime
                let static_name: &'static str = Box::leak(sym_name.to_string().into_boxed_str());

                // Read signature if available
                let sig = (*sym_ptr).get_sig().cloned();

                symbols.push((static_name, (*sym_ptr).ptr));
                symbols_with_sig.push(RuntimeSymbolInfo {
                    name: static_name,
                    ptr: (*sym_ptr).ptr,
                    sig,
                });

                sym_ptr = sym_ptr.add(1);
            }

            let sig_count = symbols_with_sig.iter().filter(|s| s.sig.is_some()).count();
            log::info!(
                "Loaded ZRTL plugin '{}' with {} symbols ({} with signatures) from {}",
                name,
                symbols.len(),
                sig_count,
                path.display()
            );

            Ok(Self {
                library,
                name,
                symbols_with_sig,
                symbols,
            })
        }
    }

    /// Get the plugin name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get all symbols exported by this plugin
    pub fn symbols(&self) -> &[(&'static str, *const u8)] {
        &self.symbols
    }

    /// Get all symbols with signature information
    pub fn symbols_with_signatures(&self) -> &[RuntimeSymbolInfo] {
        &self.symbols_with_sig
    }

    /// Get signature for a specific symbol
    pub fn get_signature(&self, name: &str) -> Option<&ZrtlSymbolSig> {
        self.symbols_with_sig
            .iter()
            .find(|s| s.name == name)
            .and_then(|s| s.sig.as_ref())
    }

    /// Check if a symbol expects DynamicBox for a parameter
    pub fn param_is_dynamic(&self, symbol_name: &str, param_index: usize) -> bool {
        self.get_signature(symbol_name)
            .map(|sig| sig.param_is_dynamic(param_index))
            .unwrap_or(false)
    }

    /// Convert to RuntimePlugin symbols format
    pub fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
        self.symbols.clone()
    }

    /// Get runtime symbols with signature info
    pub fn runtime_symbols_with_sig(&self) -> Vec<RuntimeSymbolInfo> {
        self.symbols_with_sig.clone()
    }
}

/// Errors that can occur when loading ZRTL plugins
#[derive(Debug)]
pub enum ZrtlError {
    /// Failed to load the dynamic library
    LoadFailed {
        path: std::path::PathBuf,
        reason: String,
    },
    /// Plugin is missing the _zrtl_info symbol
    MissingInfo { path: std::path::PathBuf },
    /// Plugin is missing the _zrtl_symbols symbol
    MissingSymbols { path: std::path::PathBuf },
    /// Plugin version doesn't match
    VersionMismatch {
        path: std::path::PathBuf,
        expected: u32,
        found: u32,
    },
    /// Invalid UTF-8 in plugin metadata
    InvalidUtf8 {
        path: std::path::PathBuf,
        field: String,
    },
}

impl std::fmt::Display for ZrtlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZrtlError::LoadFailed { path, reason } => {
                write!(
                    f,
                    "Failed to load ZRTL plugin '{}': {}",
                    path.display(),
                    reason
                )
            }
            ZrtlError::MissingInfo { path } => {
                write!(
                    f,
                    "ZRTL plugin '{}' is missing _zrtl_info symbol",
                    path.display()
                )
            }
            ZrtlError::MissingSymbols { path } => {
                write!(
                    f,
                    "ZRTL plugin '{}' is missing _zrtl_symbols symbol",
                    path.display()
                )
            }
            ZrtlError::VersionMismatch {
                path,
                expected,
                found,
            } => {
                write!(
                    f,
                    "ZRTL plugin '{}' version mismatch: expected {}, found {}",
                    path.display(),
                    expected,
                    found
                )
            }
            ZrtlError::InvalidUtf8 { path, field } => {
                write!(
                    f,
                    "ZRTL plugin '{}' has invalid UTF-8 in {}",
                    path.display(),
                    field
                )
            }
        }
    }
}

impl std::error::Error for ZrtlError {}

/// Registry for ZRTL plugins
pub struct ZrtlRegistry {
    plugins: Vec<ZrtlPlugin>,
}

impl ZrtlRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Load and register a ZRTL plugin from a file path
    pub fn load_plugin<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ZrtlError> {
        let plugin = ZrtlPlugin::load(path)?;
        self.plugins.push(plugin);
        Ok(())
    }

    /// Load all ZRTL plugins from a directory
    pub fn load_directory<P: AsRef<Path>>(&mut self, dir: P) -> Result<usize, ZrtlError> {
        let dir = dir.as_ref();
        let mut count = 0;

        if !dir.exists() {
            return Ok(0);
        }

        let entries = std::fs::read_dir(dir).map_err(|e| ZrtlError::LoadFailed {
            path: dir.to_path_buf(),
            reason: e.to_string(),
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(OsStr::to_str) == Some(ZRTL_EXTENSION) {
                match self.load_plugin(&path) {
                    Ok(()) => count += 1,
                    Err(e) => {
                        log::warn!("Failed to load ZRTL plugin {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(count)
    }

    /// Get all symbols from all loaded plugins
    pub fn collect_symbols(&self) -> Vec<(&'static str, *const u8)> {
        let mut symbols = Vec::new();
        for plugin in &self.plugins {
            symbols.extend(plugin.runtime_symbols());
        }
        symbols
    }

    /// Get all symbols with signature information from all loaded plugins
    pub fn collect_symbols_with_signatures(&self) -> Vec<RuntimeSymbolInfo> {
        let mut symbols = Vec::new();
        for plugin in &self.plugins {
            symbols.extend_from_slice(plugin.symbols_with_signatures());
        }
        symbols
    }

    /// List all loaded plugin names
    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }
}

impl Default for ZrtlRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// C ABI Value Conversion Functions
// ============================================================
//
// These functions are called by the Cranelift backend to convert
// between compiler values and ZRTL values.

/// Raw closure function type matching ZRTL's RawClosureFn
pub type RawClosureFn = extern "C" fn(*mut u8, i64) -> i64;

/// ZrtlClosure layout (must match sdk/zrtl/src/closure.rs)
#[repr(C)]
pub struct ZrtlClosureRepr {
    pub inner: *const (),
    pub call_fn: extern "C" fn(*const (), i64) -> i64,
    pub call_void_fn: extern "C" fn(*const (), i64),
    pub drop_fn: extern "C" fn(*const ()),
    pub clone_fn: extern "C" fn(*const ()) -> *const (),
}

/// DynamicBox layout (must match sdk/zrtl/src/dynamic_box.rs)
#[repr(C)]
pub struct DynamicBoxRepr {
    pub tag: u32,  // TypeTag
    pub size: u32, // Size in bytes
    pub data: *mut u8,
    pub dropper: Option<extern "C" fn(*mut u8)>,
    pub display_fn: Option<extern "C" fn(*const u8) -> *const u8>,
}

/// Convert compiler closure to ZrtlClosure
///
/// This creates a ZrtlClosure from a raw function pointer and environment.
/// The function signature must be `fn(env: *mut u8, arg: i64) -> i64`.
///
/// # Arguments
/// * `fn_ptr` - Function pointer (cast from closure implementation)
/// * `env_ptr` - Pointer to captured environment data
/// * `env_size` - Size of environment in bytes (0 if no captures)
///
/// # Returns
/// Pointer to heap-allocated ZrtlClosure. Caller must free with `zrtl_closure_free`.
#[no_mangle]
pub unsafe extern "C" fn zyntax_closure_to_zrtl(
    fn_ptr: RawClosureFn,
    env_ptr: *const u8,
    env_size: usize,
) -> *mut ZrtlClosureRepr {
    // Allocate and copy environment if present
    let env_copy = if env_size > 0 && !env_ptr.is_null() {
        let layout = std::alloc::Layout::from_size_align(env_size, 8).unwrap();
        let ptr = std::alloc::alloc(layout);
        std::ptr::copy_nonoverlapping(env_ptr, ptr, env_size);
        ptr
    } else {
        std::ptr::null_mut()
    };

    // Create inner raw closure wrapper
    let inner = Box::new(RawClosureWrapper {
        func: fn_ptr,
        env: env_copy,
        env_size,
    });
    let inner_ptr = Box::into_raw(inner) as *const ();

    // Create the ZrtlClosure struct
    let closure = Box::new(ZrtlClosureRepr {
        inner: inner_ptr,
        call_fn: raw_closure_call,
        call_void_fn: raw_closure_call_void,
        drop_fn: raw_closure_drop,
        clone_fn: raw_closure_clone,
    });

    Box::into_raw(closure)
}

/// Internal wrapper for raw closures
struct RawClosureWrapper {
    func: RawClosureFn,
    env: *mut u8,
    env_size: usize,
}

impl Drop for RawClosureWrapper {
    fn drop(&mut self) {
        if !self.env.is_null() && self.env_size > 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align(self.env_size, 8).unwrap();
                std::alloc::dealloc(self.env, layout);
            }
        }
    }
}

// SAFETY: Raw closures are designed to be thread-safe
unsafe impl Send for RawClosureWrapper {}
unsafe impl Sync for RawClosureWrapper {}

extern "C" fn raw_closure_call(ptr: *const (), arg: i64) -> i64 {
    unsafe {
        let wrapper = &*(ptr as *const RawClosureWrapper);
        (wrapper.func)(wrapper.env, arg)
    }
}

extern "C" fn raw_closure_call_void(ptr: *const (), arg: i64) {
    unsafe {
        let wrapper = &*(ptr as *const RawClosureWrapper);
        let _ = (wrapper.func)(wrapper.env, arg);
    }
}

extern "C" fn raw_closure_drop(ptr: *const ()) {
    unsafe {
        let _ = Box::from_raw(ptr as *mut RawClosureWrapper);
    }
}

extern "C" fn raw_closure_clone(ptr: *const ()) -> *const () {
    unsafe {
        let wrapper = &*(ptr as *const RawClosureWrapper);

        // Copy environment
        let env_copy = if wrapper.env_size > 0 && !wrapper.env.is_null() {
            let layout = std::alloc::Layout::from_size_align(wrapper.env_size, 8).unwrap();
            let new_ptr = std::alloc::alloc(layout);
            std::ptr::copy_nonoverlapping(wrapper.env, new_ptr, wrapper.env_size);
            new_ptr
        } else {
            std::ptr::null_mut()
        };

        let cloned = Box::new(RawClosureWrapper {
            func: wrapper.func,
            env: env_copy,
            env_size: wrapper.env_size,
        });

        Box::into_raw(cloned) as *const ()
    }
}

/// Free a ZrtlClosure created by zyntax_closure_to_zrtl
#[no_mangle]
pub unsafe extern "C" fn zyntax_closure_free(closure: *mut ZrtlClosureRepr) {
    if !closure.is_null() {
        let c = Box::from_raw(closure);
        (c.drop_fn)(c.inner);
    }
}

/// Convert a primitive value to a DynamicBox
///
/// # Arguments
/// * `value_ptr` - Pointer to the value data
/// * `type_tag` - ZRTL TypeTag (packed u32)
/// * `size` - Size of value in bytes
///
/// # Returns
/// Pointer to heap-allocated DynamicBox. Caller must free with `zyntax_box_free`.
#[no_mangle]
pub unsafe extern "C" fn zyntax_primitive_to_box(
    value_ptr: *const u8,
    type_tag: u32,
    size: u32,
) -> *mut DynamicBoxRepr {
    // Allocate and copy value data
    let data = if size > 0 && !value_ptr.is_null() {
        let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
        let ptr = std::alloc::alloc(layout);
        std::ptr::copy_nonoverlapping(value_ptr, ptr, size as usize);
        ptr
    } else {
        std::ptr::null_mut()
    };

    let boxed = Box::new(DynamicBoxRepr {
        tag: type_tag,
        size,
        data,
        dropper: Some(default_box_dropper),
        display_fn: None,
    });

    Box::into_raw(boxed)
}

/// Create a DynamicBox for an i32 value
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_i32(value: i32) -> *mut DynamicBoxRepr {
    let data = Box::into_raw(Box::new(value)) as *mut u8;
    let boxed = Box::new(DynamicBoxRepr {
        tag: TypeTag::I32.0,
        size: 4,
        data,
        dropper: Some(drop_box_i32),
        display_fn: None,
    });
    Box::into_raw(boxed)
}

/// Create a DynamicBox for an i64 value
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_i64(value: i64) -> *mut DynamicBoxRepr {
    let data = Box::into_raw(Box::new(value)) as *mut u8;
    let boxed = Box::new(DynamicBoxRepr {
        tag: TypeTag::I64.0,
        size: 8,
        data,
        dropper: Some(drop_box_i64),
        display_fn: None,
    });
    Box::into_raw(boxed)
}

/// Create a DynamicBox for an f32 value
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_f32(value: f32) -> *mut DynamicBoxRepr {
    let data = Box::into_raw(Box::new(value)) as *mut u8;
    let boxed = Box::new(DynamicBoxRepr {
        tag: TypeTag::F32.0,
        size: 4,
        data,
        dropper: Some(drop_box_f32),
        display_fn: None,
    });
    Box::into_raw(boxed)
}

/// Create a DynamicBox for an f64 value
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_f64(value: f64) -> *mut DynamicBoxRepr {
    let data = Box::into_raw(Box::new(value)) as *mut u8;
    let boxed = Box::new(DynamicBoxRepr {
        tag: TypeTag::F64.0,
        size: 8,
        data,
        dropper: Some(drop_box_f64),
        display_fn: None,
    });
    Box::into_raw(boxed)
}

/// Create a DynamicBox for a bool value
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_bool(value: i32) -> *mut DynamicBoxRepr {
    let data = Box::into_raw(Box::new(value as u8)) as *mut u8;
    let boxed = Box::new(DynamicBoxRepr {
        tag: TypeTag::BOOL.0,
        size: 1,
        data,
        dropper: Some(drop_box_u8),
        display_fn: None,
    });
    Box::into_raw(boxed)
}

/// Free a DynamicBox created by zyntax_box_* functions
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_free(boxed: *mut DynamicBoxRepr) {
    if !boxed.is_null() {
        let b = Box::from_raw(boxed);
        if let Some(dropper) = b.dropper {
            if !b.data.is_null() {
                dropper(b.data);
            }
        }
    }
}

/// Get the value from a DynamicBox as i64
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_get_i64(boxed: *const DynamicBoxRepr) -> i64 {
    if boxed.is_null() || (*boxed).data.is_null() {
        return 0;
    }
    match (*boxed).size {
        1 => *((*boxed).data as *const i8) as i64,
        2 => *((*boxed).data as *const i16) as i64,
        4 => *((*boxed).data as *const i32) as i64,
        8 => *((*boxed).data as *const i64),
        _ => 0,
    }
}

/// Get the TypeTag from a DynamicBox
#[no_mangle]
pub unsafe extern "C" fn zyntax_box_get_tag(boxed: *const DynamicBoxRepr) -> u32 {
    if boxed.is_null() {
        return TypeTag::VOID.0;
    }
    (*boxed).tag
}

// Default dropper for raw data
extern "C" fn default_box_dropper(ptr: *mut u8) {
    // Can't properly deallocate without knowing the layout
    // This is a fallback - typed droppers should be used
    let _ = ptr;
}

// Typed droppers
extern "C" fn drop_box_i32(ptr: *mut u8) {
    unsafe {
        let _ = Box::from_raw(ptr as *mut i32);
    }
}

extern "C" fn drop_box_i64(ptr: *mut u8) {
    unsafe {
        let _ = Box::from_raw(ptr as *mut i64);
    }
}

extern "C" fn drop_box_f32(ptr: *mut u8) {
    unsafe {
        let _ = Box::from_raw(ptr as *mut f32);
    }
}

extern "C" fn drop_box_f64(ptr: *mut u8) {
    unsafe {
        let _ = Box::from_raw(ptr as *mut f64);
    }
}

extern "C" fn drop_box_u8(ptr: *mut u8) {
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Get the TypeTag for a HIR type
///
/// This is called at compile time (during lowering) to get the
/// ZRTL TypeTag for a given type.
pub fn type_tag_for_hir_type(ty: &crate::hir::HirType) -> TypeTag {
    use crate::hir::HirType;

    match ty {
        HirType::Void => TypeTag::VOID,
        HirType::Bool => TypeTag::BOOL,
        HirType::I8 => TypeTag::I8,
        HirType::I16 => TypeTag::I16,
        HirType::I32 => TypeTag::I32,
        HirType::I64 => TypeTag::I64,
        HirType::I128 => TypeTag::new(TypeCategory::Int, 0x10, TypeFlags::NONE), // 128-bit
        HirType::U8 => TypeTag::U8,
        HirType::U16 => TypeTag::U16,
        HirType::U32 => TypeTag::U32,
        HirType::U64 => TypeTag::U64,
        HirType::U128 => TypeTag::new(TypeCategory::UInt, 0x10, TypeFlags::NONE), // 128-bit
        HirType::F32 => TypeTag::F32,
        HirType::F64 => TypeTag::F64,
        HirType::Ptr(_) => TypeTag::new(TypeCategory::Pointer, 0, TypeFlags::NONE),
        HirType::Ref { .. } => TypeTag::new(TypeCategory::Pointer, 1, TypeFlags::NONE),
        HirType::Array(_, _) => TypeTag::new(TypeCategory::Array, 0, TypeFlags::NONE),
        HirType::Vector(_, _) => TypeTag::new(TypeCategory::Array, 1, TypeFlags::NONE), // SIMD vector
        HirType::Struct(_) => TypeTag::new(TypeCategory::Struct, 0, TypeFlags::NONE),
        HirType::Union(_) => TypeTag::new(TypeCategory::Union, 0, TypeFlags::NONE),
        HirType::Function(_) => TypeTag::new(TypeCategory::Function, 0, TypeFlags::NONE),
        HirType::Closure(_) => TypeTag::new(TypeCategory::Function, 1, TypeFlags::NONE), // Closure
        HirType::Opaque(_) => TypeTag::new(TypeCategory::Opaque, 0, TypeFlags::NONE),
        HirType::ConstGeneric(_) => TypeTag::new(TypeCategory::Opaque, 1, TypeFlags::NONE),
        HirType::Generic { .. } => TypeTag::new(TypeCategory::Custom, 0, TypeFlags::NONE),
        HirType::TraitObject { .. } => TypeTag::new(TypeCategory::TraitObject, 0, TypeFlags::NONE),
        HirType::Interface { .. } => TypeTag::new(TypeCategory::TraitObject, 1, TypeFlags::NONE),
        HirType::Promise(_) => TypeTag::new(TypeCategory::Opaque, 2, TypeFlags::NONE), // Promise
        HirType::AssociatedType { .. } => TypeTag::new(TypeCategory::Opaque, 3, TypeFlags::NONE), // Associated type
        HirType::Continuation { .. } => TypeTag::new(TypeCategory::Function, 2, TypeFlags::NONE), // Continuation
        HirType::EffectRow { .. } => TypeTag::new(TypeCategory::Opaque, 4, TypeFlags::NONE), // Effect row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zrtl_symbol_layout() {
        // Verify C ABI compatibility
        // ZrtlSymbol now has 3 pointer-sized fields: name, ptr, sig
        assert_eq!(
            std::mem::size_of::<ZrtlSymbol>(),
            std::mem::size_of::<*const u8>() * 3
        );
    }

    #[test]
    fn test_zrtl_info_layout() {
        // Verify C ABI compatibility
        assert_eq!(
            std::mem::size_of::<ZrtlInfo>(),
            std::mem::size_of::<u32>() + std::mem::size_of::<*const u8>() + 4 // padding
        );
    }
}
