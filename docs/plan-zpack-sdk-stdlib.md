# ZPack & Runtime SDK: Production-Ready Standard Library Support

## Executive Summary

This plan outlines the work needed to make ZPack and the Runtime SDK robust enough for real-world native standard library development. The goal is to enable frontend implementers (ZynML, Python, Lua, or a custom language) to build comprehensive standard libraries with:

1. **Type-safe FFI** between generated code and native runtime
2. **Generic container support** (Array<T>, Map<K,V>, etc.)
3. **Memory management** that's correct and efficient
4. **Cross-platform distribution** via ZPack
5. **Developer ergonomics** with good tooling and macros

---

## Current State Analysis

### What Works

1. **Basic ZRTL loading** - `ZrtlPlugin::load()` successfully loads `.zrtl` files
2. **Symbol resolution** - Function pointers are correctly passed to JIT/AOT
3. **Basic DynamicValue** - Type tags, primitive boxing/unboxing
4. **C SDK header** - `zrtl.h` with type tags and DynamicBox
5. **Rust macros** - `zrtl_plugin!` and `#[zrtl_export]` for basic exports
6. **ZPack format** - Archive creation and loading works

### What's Missing/Incomplete

1. ~~Generic type reification~~ - ✅ EXISTS: `GenericTypeArgs`, `GenericValue` in zrtl.rs
2. **Object/struct field access** - Partial: `ZrtlFieldDescriptor` exists but no runtime reflection API
3. **Method dispatch** - No vtable or interface dispatch support in SDK
4. ~~Memory lifecycle hooks~~ - ✅ EXISTS: `DropFn`, dropper callbacks in `TypeMeta`/`DynamicBox`
5. ~~Error handling~~ - ✅ EXISTS: `ZRTL_CAT_RESULT`, `ZrtlOptional`, `zrtl_gbox_result()`
6. **Iterator protocol** - No standard iteration interface in SDK
7. **Async runtime** - Compiler has async support but SDK has no runtime hooks
8. **Debugging support** - No DWARF generation for runtime functions
9. **Testing infrastructure** - No runtime test harness or `ZRTL_TEST` macros

### Actual Gaps (Verified)

| Gap | Priority | Description |
|-----|----------|-------------|
| **Method dispatch in SDK** | Medium | No vtable support for interface calls from runtime functions |
| **Iterator protocol** | Medium | No standard `next()`/`hasNext()` interface |
| **Async runtime hooks** | Low | SDK doesn't expose async executor integration |
| **DWARF for runtime** | Low | No debug info generation for `.zrtl` functions |
| **Test harness** | Medium | No `ZRTL_TEST` macros or test runner |
| **Runtime reflection API** | Medium | Field descriptors exist but no query functions |

---

## Phase 1: Core Type System (Foundation)

### 1.1 Standardize Type Representations

**Problem**: Currently we have multiple string representations:
- Length-prefixed (`i32` length + bytes)
- Struct with ptr/len/cap

**Solution**: Define canonical representations in `zrtl.h`:

```c
// Canonical string - always heap allocated with capacity
typedef struct {
    uint8_t* data;      // UTF-8 bytes (NOT null-terminated)
    uint32_t length;    // Byte length
    uint32_t capacity;  // Allocated capacity
} ZrtlString;

// Canonical array - generic over element type
typedef struct {
    void* data;         // Element storage
    uint32_t length;    // Number of elements
    uint32_t capacity;  // Allocated slots
    uint32_t elem_size; // Sizeof one element
    ZrtlTypeTag elem_type; // Element type tag
} ZrtlArray;

// Canonical map
typedef struct {
    void* buckets;      // Hash buckets
    uint32_t length;    // Number of entries
    uint32_t capacity;  // Number of buckets
    ZrtlTypeTag key_type;
    ZrtlTypeTag value_type;
} ZrtlMap;
```

**Tasks**:
- [ ] Finalize canonical type layouts in `zrtl.h`
- [ ] Update `DynamicValue` in Rust to use same layouts
- [ ] Add conversion helpers in SDK

### 1.2 Type Registry & Reflection

**Problem**: Custom types (structs, classes) have no runtime type info.

**Solution**: Add a type registry that plugins can populate:

```rust
// In zrtl.rs
pub struct TypeRegistry {
    types: Vec<TypeDescriptor>,
    name_to_id: HashMap<String, TypeId>,
}

pub struct TypeDescriptor {
    pub name: String,
    pub size: u32,
    pub alignment: u32,
    pub dropper: Option<DropFn>,
    pub fields: Vec<FieldDescriptor>,
    pub methods: Vec<MethodDescriptor>,
}

pub struct FieldDescriptor {
    pub name: String,
    pub offset: u32,
    pub type_id: TypeId,
}

pub struct MethodDescriptor {
    pub name: String,
    pub symbol: String,  // Runtime symbol name
    pub signature: FunctionSignature,
}
```

**Tasks**:
- [ ] Implement `TypeRegistry` in `zrtl.rs`
- [ ] Add `ZRTL_REGISTER_TYPE(name, TypeDescriptor)` macro in C
- [ ] Add `#[zrtl_type]` derive macro in Rust
- [ ] Generate type registrations from frontend type info
- [ ] Support querying field offsets at runtime

### 1.3 Generic Container Protocol

**Problem**: `Array<Int>` and `Array<String>` need different handling but share code.

**Solution**: Generic containers carry element type info:

```rust
// GenericArray with element type tracking
#[repr(C)]
pub struct GenericArray {
    pub data: *mut u8,
    pub length: u32,
    pub capacity: u32,
    pub elem_size: u32,
    pub elem_type: TypeTag,
    pub elem_dropper: Option<DropFn>,  // Called on each element during cleanup
}

impl GenericArray {
    pub fn new(elem_type: TypeTag, elem_size: u32) -> Self { ... }
    pub fn push(&mut self, elem: *const u8) { ... }
    pub fn get(&self, index: u32) -> *const u8 { ... }
    pub fn drop_all(&mut self) { ... }  // Calls elem_dropper on each
}
```

**Tasks**:
- [ ] Define `GenericArray`, `GenericMap` in `zrtl.rs`
- [ ] Add C equivalents in `zrtl.h`
- [ ] Implement type-aware `push`/`get`/`set` operations
- [ ] Add element drop propagation
- [ ] Test with nested generics (`Array<Array<Int>>`)

---

## Phase 2: Memory Management

### 2.1 Ownership Model

**Problem**: No clear ownership semantics - who frees memory?

**Solution**: Adopt clear ownership rules:

1. **Runtime-allocated = Runtime-freed**: If ZRTL allocates, ZRTL frees
2. **Caller-owns-result**: Functions returning pointers transfer ownership
3. **Borrowing via const**: `const T*` means "borrowed, do not free"

Add explicit ownership markers:

```c
// Ownership annotations in function signatures
typedef struct {
    ZrtlArray* arr;  // OWNED - caller must free
} OwnedArray;

typedef struct {
    const ZrtlArray* arr;  // BORROWED - caller must NOT free
} BorrowedArray;

// Runtime function ownership convention
ZRTL_EXPORT ZrtlArray* Array_copy(const ZrtlArray* src);  // Returns owned
ZRTL_EXPORT void Array_sort_inplace(ZrtlArray* arr);       // Mutates, no ownership change
```

**Tasks**:
- [ ] Document ownership conventions in SDK
- [ ] Add `ZRTL_OWNED`, `ZRTL_BORROWED` attribute macros
- [ ] Update all runtime functions with ownership docs
- [ ] Add ownership verification in debug builds

### 2.2 Destructor Chain

**Problem**: Nested structures need recursive cleanup.

**Solution**: Type descriptors include drop functions that handle children:

```rust
// Drop chain example for Array<String>
extern "C" fn drop_string_array(arr: *mut GenericArray) {
    unsafe {
        let arr = &mut *arr;
        // Drop each string element
        for i in 0..arr.length {
            let elem_ptr = arr.get(i) as *mut ZrtlString;
            if !elem_ptr.is_null() {
                drop_zrtl_string(elem_ptr);
            }
        }
        // Free array storage
        libc::free(arr.data as *mut _);
    }
}
```

**Tasks**:
- [ ] Add `dropper` field to all container types
- [ ] Generate drop functions for parameterized types
- [ ] Handle cycles (weak refs) to prevent leaks
- [ ] Add debug mode leak detection

### 2.3 Reference Counting Integration

**Problem**: Some objects need shared ownership (closures, callbacks).

**Solution**: Optional ARC wrapper for shared values:

```rust
#[repr(C)]
pub struct ArcHeader {
    pub strong_count: AtomicU32,
    pub weak_count: AtomicU32,
    pub dropper: DropFn,
}

#[repr(C)]
pub struct ArcBox<T> {
    header: ArcHeader,
    value: T,
}

// C API
ZRTL_EXPORT void* zrtl_arc_new(size_t size, ZrtlDropFn dropper);
ZRTL_EXPORT void zrtl_arc_retain(void* arc_ptr);
ZRTL_EXPORT void zrtl_arc_release(void* arc_ptr);
ZRTL_EXPORT void* zrtl_arc_weak_upgrade(void* weak_ptr);
```

**Tasks**:
- [ ] Implement `ArcBox` in `zrtl.rs`
- [ ] Add C wrappers in `zrtl.h`
- [ ] Integrate with frontend's object model
- [ ] Handle weak references for cycles

---

## Phase 3: Method Dispatch

### 3.1 Direct Method Calls

**Problem**: Instance methods need receiver + method resolution.

**Solution**: Method symbols encode receiver type:

```
Symbol naming: $<Type>$<method>
Examples:
  $Array$push       - Array.push(elem)
  $String$length    - String.length property
  $MyClass$doThing  - MyClass.doThing()
```

Compiler generates calls:
```llvm
; obj.push(42) where obj: Array<Int>
%result = call i32 @"$Array$push"(%Array* %obj, i32 42)
```

**Tasks**:
- [ ] Document method symbol naming convention
- [ ] Generate correct symbols from frontend
- [ ] Support property getter/setter disambiguation

### 3.2 Virtual Method Dispatch (Interfaces/Traits)

**Problem**: Interface calls need vtable lookup.

**Solution**: Vtables for interface types:

```rust
// Vtable structure for an interface
#[repr(C)]
pub struct IteratorVtable {
    pub next: extern "C" fn(*mut u8) -> DynamicValue,
    pub has_next: extern "C" fn(*const u8) -> bool,
}

// Trait object representation
#[repr(C)]
pub struct TraitObject {
    pub data: *mut u8,
    pub vtable: *const (),
}
```

**Tasks**:
- [ ] Define vtable layout convention
- [ ] Generate vtables for implementing types
- [ ] Add `TraitObject` to ZRTL
- [ ] Implement interface dispatch in HIR lowering

### 3.3 Dynamic Dispatch (for dynamically typed values)

**Problem**: a dynamically typed value needs runtime method lookup.

**Solution**: Method table attached to object metadata:

```rust
// Object header for Dynamic-capable objects
#[repr(C)]
pub struct ObjectHeader {
    pub type_id: TypeId,
    pub method_table: *const MethodTable,
}

// Method lookup table
pub struct MethodTable {
    pub entries: Vec<MethodEntry>,
}

pub struct MethodEntry {
    pub name_hash: u64,
    pub name: &'static str,
    pub ptr: *const u8,
}

// Runtime lookup
extern "C" fn dynamic_method_lookup(
    obj: *const ObjectHeader,
    name: *const c_char,
) -> *const u8;
```

**Tasks**:
- [ ] Implement `ObjectHeader` and `MethodTable`
- [ ] Add method table generation to frontend
- [ ] Implement `dynamic_method_lookup` function
- [ ] Cache lookups for performance

---

## Phase 4: Standard Library Primitives

### 4.1 String Operations

```rust
// Complete string API
pub trait ZrtlStringOps {
    fn new_empty() -> ZrtlString;
    fn from_utf8(bytes: &[u8]) -> ZrtlString;
    fn from_cstr(cstr: *const c_char) -> ZrtlString;
    fn length(&self) -> u32;
    fn char_at(&self, index: u32) -> u32;  // Returns codepoint
    fn substring(&self, start: u32, end: u32) -> ZrtlString;
    fn concat(&self, other: &ZrtlString) -> ZrtlString;
    fn split(&self, delimiter: &ZrtlString) -> GenericArray;
    fn to_lowercase(&self) -> ZrtlString;
    fn to_uppercase(&self) -> ZrtlString;
    fn trim(&self) -> ZrtlString;
    fn index_of(&self, needle: &ZrtlString) -> i32;
    fn replace(&self, from: &ZrtlString, to: &ZrtlString) -> ZrtlString;
}
```

**Tasks**:
- [ ] Implement all string ops in `zrtl_string.rs`
- [ ] Export as `$String$*` symbols
- [ ] Handle Unicode correctly (grapheme clusters)
- [ ] Add string interning for small strings

### 4.2 Math Operations

```rust
// Math library
extern "C" fn math_sin(x: f64) -> f64;
extern "C" fn math_cos(x: f64) -> f64;
extern "C" fn math_sqrt(x: f64) -> f64;
extern "C" fn math_pow(base: f64, exp: f64) -> f64;
extern "C" fn math_floor(x: f64) -> f64;
extern "C" fn math_ceil(x: f64) -> f64;
extern "C" fn math_round(x: f64) -> f64;
extern "C" fn math_random() -> f64;
extern "C" fn math_abs_int(x: i64) -> i64;
extern "C" fn math_abs_float(x: f64) -> f64;
```

**Tasks**:
- [ ] Implement math functions (can wrap libc)
- [ ] Export as `$Math$*` symbols
- [ ] Handle edge cases (NaN, Inf)

### 4.3 I/O Operations

```rust
// File I/O
extern "C" fn file_open(path: *const ZrtlString, mode: i32) -> *mut FileHandle;
extern "C" fn file_read(handle: *mut FileHandle, buf: *mut u8, len: u32) -> i32;
extern "C" fn file_write(handle: *mut FileHandle, buf: *const u8, len: u32) -> i32;
extern "C" fn file_close(handle: *mut FileHandle);

// Console I/O
extern "C" fn console_print(s: *const ZrtlString);
extern "C" fn console_println(s: *const ZrtlString);
extern "C" fn console_read_line() -> ZrtlString;
```

**Tasks**:
- [ ] Implement file ops with error handling
- [ ] Console I/O with buffering
- [ ] Export as `$Sys$*` / `$File$*` symbols

### 4.4 Date/Time

```rust
extern "C" fn date_now() -> i64;  // Unix timestamp ms
extern "C" fn date_from_timestamp(ts: i64) -> *mut DateTime;
extern "C" fn date_year(dt: *const DateTime) -> i32;
extern "C" fn date_month(dt: *const DateTime) -> i32;
// ... etc
```

**Tasks**:
- [ ] Implement DateTime type
- [ ] Time zone handling
- [ ] Formatting and parsing

---

## Phase 5: SDK & Tooling

### 5.1 Enhanced Rust Macros

```rust
// Type definition macro
#[zrtl_struct("MyPoint")]
#[derive(Clone)]
pub struct MyPoint {
    pub x: f64,
    pub y: f64,
}

// Auto-generates:
// - TypeDescriptor registration
// - Field accessors as symbols
// - Serialization hooks

// Method export with full metadata
#[zrtl_method(
    symbol = "$MyPoint$distance",
    receiver = "MyPoint",
    args = [("other", "MyPoint")],
    returns = "f64"
)]
pub extern "C" fn my_point_distance(self_: *const MyPoint, other: *const MyPoint) -> f64 {
    // ...
}
```

**Tasks**:
- [ ] Implement `#[zrtl_struct]` derive macro
- [ ] Implement `#[zrtl_method]` with metadata
- [ ] Generate type descriptors automatically
- [ ] Support generic type parameters

### 5.2 C SDK Improvements

```c
// Enhanced C macros
ZRTL_DEFINE_STRUCT(MyPoint,
    ZRTL_FIELD(x, f64),
    ZRTL_FIELD(y, f64)
);

ZRTL_METHOD(MyPoint, distance, f64,
    ZRTL_ARG(other, MyPoint*)
) {
    return sqrt(pow(self->x - other->x, 2) + pow(self->y - other->y, 2));
}
```

**Tasks**:
- [ ] Add struct definition macros
- [ ] Add method definition macros
- [ ] Auto-generate symbol table from macros
- [ ] Add compile-time type checking where possible

### 5.3 ZPack CLI Commands

```bash
# Pack a runtime
zyntax pack create \
    --name mylang-std \
    --version 1.0.0 \
    --language mylang \
    --runtime lib/darwin-arm64/runtime.zrtl \
    --runtime lib/linux-x64/runtime.zrtl \
    --output mylang-std.zpack

# Inspect a pack
zyntax pack info mylang-std.zpack

# Extract contents
zyntax pack extract mylang-std.zpack --output ./extracted/

# List symbols
zyntax pack symbols mylang-std.zpack

# Verify pack integrity
zyntax pack verify mylang-std.zpack
```

**Tasks**:
- [ ] Implement `zyntax pack create` command
- [ ] Implement `zyntax pack info` command
- [ ] Implement `zyntax pack extract` command
- [ ] Implement `zyntax pack symbols` command
- [ ] Implement `zyntax pack verify` command

### 5.4 Symbol Verification Tool

```bash
# Check if a runtime provides required symbols
zyntax symbols check \
    --required ./generated/symbols.txt \
    --runtime ./runtime.zrtl

# Generate symbol requirements from compiled program
zyntax symbols list \
    --source ./compiled.zbc \
    --output ./required-symbols.txt
```

**Tasks**:
- [ ] Extract required symbols from bytecode
- [ ] Verify runtime provides all required symbols
- [ ] Report missing symbols with suggestions

---

## Phase 6: Testing & Quality

### 6.1 Runtime Test Harness

```rust
// In sdk/zrtl_test/
pub struct TestRunner {
    tests: Vec<TestCase>,
}

pub struct TestCase {
    name: String,
    test_fn: extern "C" fn() -> bool,
}

// C API for test registration
ZRTL_TEST(test_array_push) {
    ZrtlArray* arr = Array_new();
    arr = Array_push(arr, 42);
    ZRTL_ASSERT_EQ(Array_length(arr), 1);
    ZRTL_ASSERT_EQ(Array_get(arr, 0), 42);
    Array_free(arr);
    return true;
}

ZRTL_TEST_MAIN()  // Generates main that runs all tests
```

**Tasks**:
- [ ] Create `zrtl_test` crate
- [ ] Implement test registration macros
- [ ] Add assertion macros
- [ ] Add memory leak detection in tests
- [ ] CI integration

### 6.2 Fuzzing Infrastructure

```bash
# Fuzz test array operations
cargo fuzz run array_ops -- -max_len=1000

# Fuzz string parsing
cargo fuzz run string_parse
```

**Tasks**:
- [ ] Set up cargo-fuzz for runtime
- [ ] Create fuzz targets for each data type
- [ ] Add sanitizers (ASan, UBSan)
- [ ] Continuous fuzzing in CI

### 6.3 Benchmark Suite

```rust
// Benchmark comparisons
fn bench_array_push(c: &mut Criterion) {
    c.bench_function("array_push_1000", |b| {
        b.iter(|| {
            let mut arr = Array_new();
            for i in 0..1000 {
                arr = Array_push(arr, i);
            }
            Array_free(arr);
        });
    });
}
```

**Tasks**:
- [ ] Set up criterion benchmarks
- [ ] Benchmark all core operations
- [ ] Track regressions in CI

---

## Phase 7: Documentation

### 7.1 SDK Reference

- Complete API documentation for `zrtl.h`
- Rust crate documentation with examples
- Memory management guide
- Type system reference

### 7.2 Tutorial: Building a Runtime

Step-by-step guide covering:
1. Project setup
2. Implementing basic types
3. Adding methods
4. Memory management
5. Testing
6. Packaging
7. Distribution

### 7.3 Architecture Documentation

- Type representation diagrams
- Memory layout documentation
- Symbol naming conventions
- Vtable structure

---

## Implementation Priority

### Must Have (MVP)
1. Standardized type representations (Phase 1.1)
2. Generic container protocol (Phase 1.3)
3. Destructor chain (Phase 2.2)
4. Direct method calls (Phase 3.1)
5. String operations (Phase 4.1)

### Should Have
1. Type registry & reflection (Phase 1.2)
2. Ownership model documentation (Phase 2.1)
3. Reference counting (Phase 2.3)
4. Math operations (Phase 4.2)
5. Enhanced Rust macros (Phase 5.1)

### Nice to Have
1. Virtual dispatch (Phase 3.2)
2. Dynamic dispatch (Phase 3.3)
3. I/O operations (Phase 4.3)
4. Date/Time (Phase 4.4)
5. Full tooling suite (Phase 5.3-5.4)

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 | 2-3 weeks | None |
| Phase 2 | 2 weeks | Phase 1 |
| Phase 3.1 | 1 week | Phase 1 |
| Phase 3.2-3.3 | 2 weeks | Phase 3.1 |
| Phase 4 | 3-4 weeks | Phase 1-2 |
| Phase 5 | 2-3 weeks | Phase 1-4 |
| Phase 6 | 2 weeks | Phase 4-5 |
| Phase 7 | Ongoing | All |

**Total: ~14-17 weeks for full implementation**

---

## Success Criteria

1. **A frontend's standard library compiles and runs** - All basic types work
2. **No memory leaks in test suite** - Valgrind/ASan clean
3. **Cross-platform builds** - macOS, Linux, Windows all work
4. **Performance parity** - Within 2x of HashLink for benchmarks
5. **Developer adoption** - At least one external frontend uses SDK
