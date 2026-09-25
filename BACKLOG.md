# Zyntax Compiler - Development Backlog

**Last Updated**: November 27, 2025 (GPU section revised 2026-09-25)
**Current Status**: Production-Ready Core (100% tests passing, 71/71 Zig tests)

**Recent Progress**:

- ✅ **Switch Expression Pattern Matching Complete** (November 27, 2025)
  - Multi-case switch with literal patterns and else clause
  - Tagged union patterns (`.some`, `.none`) with graceful non-enum handling
  - Error patterns (`error.OutOfMemory`)
  - Grammar support for all pattern types (literal, wildcard, range, struct, enum, array, pointer, error)
  - Documentation updated (`book/05-semantic-actions.md`, `book/08-zig-example.md`)
- ✅ **LLVM Backend Working** (AOT and JIT modes fully functional - functions, structs, generics)
  - Parameter value mapping for function calls
  - Struct initialization with InsertValue
  - Generic function instantiation with comptime
- ✅ **Comptime Parameter Support** (Zig grammar: `fn foo(comptime T: type, x: T) T`)
- ✅ **ZynPEG VM Runtime** (No Rust compilation required! JSON command pattern for dynamic AST construction)
  - `pest_vm` for dynamic grammar parsing
  - JSON action blocks in `.zyn` files → runtime interpretation
  - `CommandInterpreter` executes zpeg commands via host functions
  - Full pipeline: `.zyn` grammar → `.zpeg` module → parse source → TypedAST → HIR → JIT
- ✅ **ZynPEG Grammar Specification** (`docs/ZYN_GRAMMAR_SPEC.md` - Version 2.0 with JSON action blocks)
- ✅ **ZynPEG Integration Test with JIT** (Full pipeline: ZynPEG grammar → pest + AST builder → TypedAST → HIR → Cranelift JIT → Execution)
- ✅ **Grammar Conventions Documentation** (`crates/zyn_peg/GRAMMAR_CONVENTIONS.md` - best practices for .zyn files)
- ✅ **Fixed fn_params bug** (all_children iteration instead of .find() to collect ALL parameters)
- ✅ **Fixed binary expressions** (fold_binary pattern for proper left-associative parsing)
- ✅ **Try expressions in loops fixed** (SSA continuation_block handling for Jump terminators)
- ✅ **Generic function monomorphization** (comptime type params, type_args extraction, full pipeline)
- ✅ **Pointer JIT execution** (address-taken variables allocated on stack, full load/store support)
- ✅ **Address-of operator** (`&expr` creates Reference expression)
- ✅ **Pointer dereference** (`ptr.*` creates Dereference expression)
- ✅ **Compound assignment operators** (`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`)
- ✅ **Null/undefined literals** (parsing support for optional type values)
- ✅ **Union declarations** (grammar: `union { }` and `union(enum) { }`)
- ✅ **Error set declarations** (grammar: `error { NotFound, ... }`)
- ✅ **Lambda/closure expressions** (full closure support with captured variables)
- ✅ **Indirect function calls** (call_indirect via function pointers)
- ✅ **Bitwise operators** (`&`, `|`, `^`, `<<`, `>>`, `~`)
- ✅ **Defer/errdefer statements** (grammar support)
- ✅ **Orelse/catch operators** (error handling operators)
- ✅ **Enum declarations** (grammar support)
- ✅ **Try expression / error propagation** (full try operator implementation)
- ✅ **Switch expressions working** (test_zig_jit_switch_expression passing)
- ✅ **Fixed array indexing in loops bug** (SSA phi node cycle breaking)
- ✅ **Pattern matching infrastructure complete** (Some/Ok/Err working)
- ✅ All loop tests passing (for, while, continue, break)
- ✅ Lambda tests passing (basic lambda, capture, calls in loops)
- ✅ **71/71 Zig E2E JIT tests passing (100%)**

---

## 🎯 Active Development

### 0. ZynPEG Parser Generator (HIGHEST PRIORITY) ✅ PHASE 3 COMPLETE

**Goal**: Build multi-language frontend infrastructure using PEG parser generator

**Status**: Phase 1-3 Complete, Phase 4 (Multi-language) Planned

**Phase 1: Calculator POC** ✅ **COMPLETE**
- [x] Create `crates/zyn_parser` workspace crate
- [x] Integrate pest PEG parser (v2.7)
- [x] Implement calculator.pest grammar
- [x] Build TypedAST generator (pest parse trees → TypedAST)
- [x] Integrate with lowering API (`LoweringContext::lower_program()`)
- [x] End-to-end JIT execution tests (16/16 passing - 100%)
- [x] **Bug Fix**: Fixed I32 literal translation in `ssa.rs` (was incorrectly using I64)
- [x] Documentation: [Phase 1 Completion Report](docs/ZYN_PARSER_PHASE1_COMPLETION.md)

**Key Achievement**: Demonstrated full **pest → TypedAST → HIR → JIT** pipeline ✅

**Test Results**:
```
✅ 8 unit tests (grammar parsing + TypedAST building)
✅ 3 E2E JIT execution tests ("2 + 3" = 5, "(2 + 3) * 4" = 20, etc.)
✅ 5 TypedAST validation tests
```

**Phase 2: Zig Subset** ✅ **COMPLETE**
- [x] Implement zig.pest grammar (350+ lines) ✅
- [x] Support: structs, functions, control flow ✅
- [x] Variables, operators, type system ✅
- [x] Logical operators with short-circuit evaluation ✅
- [x] Continue/break statements ✅
- [x] **Array literals and array types** ✅
- [x] **Array indexing in loops** (fixed SSA phi node bug) ✅
- [x] **String literals** (basic support, lowered to `*i8` globals) ✅
- [x] **Optional types** (`?T` syntax) ✅
- [x] **Pattern matching** (Some/Ok/Err variants working) ✅
- [x] **Switch expressions** (full support with literal and wildcard patterns) ✅
- [x] **Try expression** (error propagation with auto-wrapping Result returns) ✅
- [x] **Lambda/closure expressions** (with captured variables) ✅
- [x] **Indirect function calls** (call_indirect for function pointers) ✅
- [x] **Bitwise operators** (`&`, `|`, `^`, `<<`, `>>`, `~`) ✅
- [x] **Defer/errdefer statements** (grammar support) ✅
- [x] **Orelse/catch operators** (error handling) ✅
- [x] **Enum declarations** (grammar support) ✅
- [x] **Array index assignment** (arr[i] = value syntax) ✅
- [x] **Compound assignment operators** (`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`) ✅
- [x] **Null/undefined literals** (parsing support) ✅
- [x] **Union declarations** (grammar: `union { }` and `union(enum) { }`) ✅
- [x] **Error set declarations** (grammar: `error { NotFound, ... }`) ✅
- [x] **Generic functions** (parsing + monomorphization complete) ✅
- [x] Generic call site type inference ✅
- [x] **Address-of operator** (`&expr` creates reference) ✅
- [x] **Pointer dereference** (`ptr.*` dereferences pointer) ✅
- [x] **Pointer JIT execution** (SSA stack allocation for address-taken vars) ✅
- [x] 71/71 E2E JIT tests passing (100%) ✅
- [x] Documentation: [Phase 2 Plan](docs/ZYN_PARSER_PHASE2_PLAN.md)

**Phase 3: TypedAST Integration** ✅ **COMPLETE**

- [x] ZynPEG grammar generates pest + AST builder code
- [x] Generated code builds `zyntax_typed_ast::TypedProgram` directly
- [x] Full JIT compilation test (parse → TypedAST → HIR → Cranelift → Execute)
- [x] `add(10, 20) = 30` works in integration test
- [x] **Grammar Conventions Documentation** (`crates/zyn_peg/GRAMMAR_CONVENTIONS.md`)
- [x] Fixed `fn_params` bug (use `all_children.filter()` not `.find()`)
- [x] Fixed binary expression rules (use `fold_binary` pattern)
- [x] **NOTE**: JIT tests require `--test-threads=1` (Cranelift not thread-safe)

**Phase 3.5: ZynPEG VM Runtime** ✅ **COMPLETE**

No Rust compilation required for grammar users! Dynamic interpretation of JSON action blocks.

- [x] **JSON Action Block Syntax** - `.zyn` grammars use JSON commands instead of Rust code
- [x] **ZPEG Module Format** - Compiled `.zpeg` contains pest grammar + JSON command mappings
- [x] **pest_vm Integration** - Dynamic grammar parsing at runtime (no pest_derive compilation)
- [x] **CommandInterpreter** - Executes zpeg commands via host functions
- [x] **AstHostFunctions Trait** - Pluggable AST construction interface
- [x] **TypedAstBuilder** - Uses `TypedASTBuilder` fluent API from `zyntax_typed_ast`
- [x] **CLI Integration** - `zyntax compile --grammar foo.zyn --source code.lang --format zyn`
- [x] **Grammar Specification** - `docs/ZYN_GRAMMAR_SPEC.md` (Version 2.0)
- [x] **End-to-end JIT execution** - Calculator example returns correct value (42)!

**Architecture**:
```
┌─ Compile Time ────────────────────────────────────────┐
│  .zyn grammar  →  ZynPEG Compiler  →  .zpeg module   │
│                                                       │
│  .zpeg contains:                                     │
│  - pest grammar (string)                             │
│  - Rule command mappings (JSON)                      │
└───────────────────────────────────────────────────────┘

┌─ Runtime ─────────────────────────────────────────────┐
│  source.lang + .zpeg  →  pest_vm  →  Commands  → AST │
│                                                       │
│  No Rust compilation required!                       │
└───────────────────────────────────────────────────────┘
```

**Remaining Items** (deferred):

- [ ] Binary expression folding - `fold_binary` command for proper left-associative arithmetic
- [ ] Pattern matching - None literal (arena symbol resolution issue)
- [ ] String operations (needs stdlib integration via plugin system)

**Documents**:
- [Grammar Specification](docs/ZYN_GRAMMAR_SPEC.md) ← NEW (Version 2.0 JSON format)
- [Implementation Plan](docs/ZYN_PARSER_IMPLEMENTATION.md)
- [Phase 1 Completion](docs/ZYN_PARSER_PHASE1_COMPLETION.md)
- [Phase 2 Plan](docs/ZYN_PARSER_PHASE2_PLAN.md)
- [Parser Library Comparison](docs/PARSER_LIBRARY_COMPARISON.md)

---

### 1. Reflaxe/Haxe Integration (HIGH PRIORITY) 🚧 IN PROGRESS
**Goal**: Create `reflaxe.Zyntax` backend to tap into Haxe's mature ecosystem

**Status**: Phase 1 Complete, Phase 2 In Progress

**Completed Tasks**:
- [x] Set up Reflaxe project structure (`reflaxe.zyntax/`)
- [x] Implement Haxe Typed AST → Zyntax TypedAST JSON translator
  - [x] `compileClassImpl()` - Classes to JSON
  - [x] `compileEnumImpl()` - Enums to JSON
  - [x] `compileExpressionImpl()` - Full expression support
- [x] Create Zyntax CLI (`crates/zyntax_cli/`) for JSON compilation
- [x] Documentation: [HAXE_INTEGRATION.md](docs/HAXE_INTEGRATION.md)
- [x] Test setup with HelloWorld example

**Current Phase (2/4): TypedAST JSON → HIR Conversion**
- [ ] Implement `typed_ast_to_hir()` in CLI
  - [ ] Function declarations → HirFunction
  - [ ] Expressions → HIR instructions
  - [ ] Type mapping (TypedAST → HIR types)
  - [ ] Control flow translation
- [ ] Test with simple functions (add, fibonacci)

**Remaining Phases**:
- **Phase 3**: Standard library mapping
  - [ ] Haxe stdlib → Zyntax runtime bridge
  - [ ] String/Array/Map implementations
  - [ ] I/O and file operations
- **Phase 4**: Advanced features
  - [ ] Generics instantiation
  - [ ] Closures with captures
  - [ ] Async/await integration
  - [ ] Performance benchmarking

**Benefits**:
- Instant access to thousands of production-tested libraries
- Validation at scale with real-world code
- Positioning as high-performance backend for Haxe

---

## Compiler Infrastructure

### 2. LLVM AOT Backend (MEDIUM PRIORITY) ✅ CORE COMPLETE

**Status**: Working for core features (functions, structs, generics)
**Location**: `crates/compiler/src/llvm_backend.rs`

**Completed (November 27, 2025)**:

- [x] Function calls with parameters (SSA parameter value mapping)
- [x] Struct initialization (InsertValue instruction)
- [x] Struct field access (ExtractValue instruction)
- [x] Generic function instantiation with comptime
- [x] Both AOT and JIT modes working
- [x] Basic control flow (if/while)
- [x] Arithmetic operations

**Remaining Work**:

- [ ] Advanced control flow (switch, pattern matching)
- [ ] Error union types (try/catch)
- [ ] Implement optimization passes
- [ ] Cross-platform testing

### 3. GPU Kernels (MEDIUM PRIORITY)
**Status (2026-09-25)**: GPU code generation is not built. The Q1-Q2 2026 plan passed with no GPU work, and the `compute` feature flag it named was never added (`cargo build --features compute` fails). The direction was revised on 2026-09-25: `@kernel` is a Zyntax capability every frontend reaches through a `zyntax` module (ZynML natively, Python `from zyntax import kernel`, a Lua `zyntax` table); kernels are a typed subset with no dynamic fallback; Metal comes first on the Mac, with Accelerate/MPS for GEMM-shaped kernels; NVIDIA goes through LLVM NVPTX and a `cudarc` driver binding.

**Built**: ZynML `compute()` grammar, `@kernel elementwise` for one in-place shape and a partial `@kernel reduce`, on CPU SIMD through HIR vectors on all four backends.

**Design**:
- [GPU AOT Architecture](docs/GPU_AOT_ARCHITECTURE.md) - NVIDIA design, status and direction
- [GPU Compute System](docs/ml-dsl-plans/09-gpu-compute-system.md) - kernel surface, Metal design, benchmark plan
- [ZynML Unified DSL](docs/ml-dsl-plans/00-unified-ml-dsl.md) - DSL with compute() target syntax

**Work items** live in git-bug (`git-bug bug show <id>`):
- `13983a22693b250bbf4b91f5dca193b3ba0df4bbc094722791e0d3c7cf170811` unmatched `compute()` bodies call an undefined runtime function
- `ab79beb59e7f172037869210d65eb668770a80c9099ddfc7dc373e3d6e796988` `@kernel reduce` returns the last yield
- `a127ddefe45d10932af8f3b2d4816181ce0d2d40ba3ddcc6465976286d00b2bf` compute modifiers are parsed and never read
- `77df2243b3427cedf9c0e9c5f79f88efd67a61f842ca54ba845f44065c95e3cb` elementwise kernel loop hard-codes 4 lanes
- `4eb2f34e13c8a37737b22e0b241a03d66d5deaee863217dd5c5efe85c25c02f5` GPU backends and the `zyntax` kernel module

### 4. Bytecode Interpreter (LOW PRIORITY)
**Status**: Specification complete
**Location**: `docs/BYTECODE_FORMAT_SPEC.md`

**Remaining Work**:
- [ ] Implement bytecode VM
- [ ] Serialization/deserialization
- [ ] JIT fallback integration

---

## Language Features

### 4. Error Handling

#### ✅ Zig-Style Error Handling (COMPLETE)
Value-based error handling without stack unwinding. Fully working with E2E tests.

**Completed**:
- [x] Error union types: `!T` (function returns error or value)
- [x] `try` expression (error propagation, auto-return on error)
- [x] `try` in loops (SSA continuation_block handling)
- [x] Chained try expressions
- [x] `catch` operator (handle error with default/handler)
- [x] `orelse` operator (unwrap optional with default)
- [x] Error set declarations: `const FileError = error { NotFound, ... };`

**E2E Tests Passing**:
- `test_zig_jit_try_expression`
- `test_zig_jit_try_error_propagation`
- `test_zig_jit_try_in_loop`
- `test_zig_jit_chained_try`
- `test_zig_jit_catch_operator`
- `test_zig_jit_orelse_operator`
- `test_zig_jit_error_union_type`

#### 🔄 Haxe-Style Exception Handling (PENDING)
Stack-unwinding exceptions for Haxe/Java-style languages. Required for Reflaxe integration.

**Tasks**:
- [ ] Design exception model (throw/catch/finally with stack unwinding)
- [ ] Implement throw expression in parser
- [ ] Add exception types to type system
- [ ] Generate exception handling HIR (landing pads, cleanup)
- [ ] Cranelift backend support for stack unwinding
- [ ] Integration with Haxe's try/catch semantics

### 5. Pattern Matching ✅ COMPLETE
**Status**: Fully working with E2E tests

**Completed**:
- [x] Basic pattern matching syntax (if-let)
- [x] Union type construction (Some, Ok, Err)
- [x] Discriminant checking and branching
- [x] Pattern variable extraction
- [x] Match arm body extraction with return statements
- [x] Switch expressions with literal and wildcard patterns
- [x] `test_pattern_match_runtime_execution` passing (Some/None)
- [x] `test_zig_jit_switch_expression` passing
- [x] **Multi-case switch expressions** (multiple literal cases with else)
- [x] **Tagged union patterns** (`.some`, `.none` - graceful handling for non-enum scrutinee)
- [x] **Error patterns** (`error.OutOfMemory`)
- [x] **Grammar support for all pattern types**:
  - `literal_pattern` - Match exact values
  - `wildcard_pattern` - Match anything (`_` or `else`)
  - `range_pattern` - Match value ranges (`1..10`)
  - `identifier_pattern` - Bind matched value to variable
  - `struct_pattern` - Match struct fields
  - `field_pattern` - Match individual struct field
  - `enum_pattern` - Match tagged union variant
  - `array_pattern` - Match array elements
  - `pointer_pattern` - Match through pointer dereference
  - `error_pattern` - Match error values
- [x] **Documentation updated** (`book/05-semantic-actions.md`, `book/08-zig-example.md`)

**Known Issue** (minor):

- [ ] None literal (arena symbol resolution - Symbol 7 doesn't resolve)
  - Root cause: Parser and SSA arena symbol lookup mismatch
  - Impact: Minor - workaround exists using if-let on None branch

**Pending Features**:

- [ ] Or patterns (`Some(x) | None`)
- [ ] Pattern guards (`if` conditions)
- [ ] Slice patterns (`[first, .., last]`)

---

## Standard Library

### 6. Extended Collections
**Status**: Core collections complete (Vec, String, HashMap)

**Additions Needed**:
- [ ] HashSet<T>
- [ ] BTreeMap<K,V> / BTreeSet<T>
- [ ] LinkedList<T>
- [ ] VecDeque<T>

### 7. I/O and File System
**Status**: Not started

**Tasks**:
- [ ] File operations (read, write, open, close)
- [ ] Directory operations
- [ ] Path manipulation
- [ ] Stdio (stdin, stdout, stderr)

### 8. Networking
**Status**: Not started

**Tasks**:
- [ ] TCP sockets
- [ ] UDP sockets
- [ ] HTTP client/server (basic)
- [ ] Async networking integration

---

## Testing & Quality

### 9. Integration Testing
**Status**: 280/284 tests passing

**Remaining Work**:
- [ ] Fix 4 failing tests
- [ ] Add more real-world test cases
- [ ] Performance regression testing
- [ ] Continuous integration improvements

### 10. Fuzzing & Property Testing
**Status**: Not started

**Tasks**:
- [ ] Set up fuzzing infrastructure
- [ ] Property-based testing framework
- [ ] Continuous fuzzing in CI

---

## Performance

### 11. Tiered JIT Optimization
**Status**: Framework complete, needs tuning

**Tasks**:
- [ ] Profile-guided optimization (PGO)
- [ ] Better hot-path detection
- [ ] Deoptimization support
- [ ] Inline caching

### 12. Memory Management
**Status**: Basic ownership works

**Enhancements**:
- [ ] Escape analysis optimization
- [ ] Stack allocation promotion
- [ ] Better lifetime elision
- [ ] Reference counting optimization

---

## Tooling

### 13. Language Server Protocol (LSP)
**Status**: Not started

**Tasks**:
- [ ] Implement LSP server
- [ ] Autocomplete
- [ ] Go-to-definition
- [ ] Refactoring support
- [ ] IDE integration (VSCode, IntelliJ)

### 14. Package Manager
**Status**: Not started

**Design Considerations**:
- [ ] Package registry
- [ ] Dependency resolution
- [ ] Build system integration
- [ ] Version management

---

## Documentation

### 15. Language Reference
**Status**: Needs comprehensive docs

**Required Documentation**:
- [ ] Language syntax reference
- [ ] Type system guide
- [ ] Async/await tutorial
- [ ] FFI guide
- [ ] Standard library API docs

### 16. Tutorials & Examples
**Status**: Limited examples

**Needed**:
- [ ] "Getting Started" guide
- [ ] Common patterns cookbook
- [ ] Real-world application examples
- [ ] Performance tuning guide

---

## Research & Exploration

### 17. Effect System
**Status**: Research phase

**Exploration**:
- [ ] Effect type design
- [ ] Effect handlers
- [ ] Integration with async/await
- [ ] Algebraic effects

### 18. Dependent Types (Advanced)
**Status**: Basic framework exists

**Enhancements**:
- [ ] Refinement types
- [ ] Indexed families
- [ ] Full dependent function types
- [ ] Proof obligations

---

## Completed (Archive)

### 🚧 Generic Function Parsing (November 18, 2025) - Phase 1 Complete
**Goal**: Support generic functions with comptime type parameters
**Status**: Parsing complete, monomorphization integration pending

**Completed**:
- Added `type_params: Vec<TypedTypeParam>` field to TypedFunction
- Grammar support for `comptime T: type` parameters
- Builder parses type params and adds to scope as TypeVar
- Test infrastructure in place (test_zig_jit_generic_function)

**Pending**:
- Connect existing monomorphization engine (monomorphize.rs)
- Type argument separation at call sites
- Generic instantiation in lowering/SSA

**Note**: Monomorphization engine already exists and passes all 11 tests

### ✅ Try Expression / Error Propagation (November 18, 2025)
**Goal**: Implement Zig's `try` operator for error union types (`!T`)
**Problem**: `try` expression test failing - returned 0 instead of 42
**Root Causes**:
1. Functions returning `!T` weren't wrapping plain return values in Result::Ok union
2. ExtractValue instruction had wrong `ty` field (aggregate type instead of extracted value type)
3. Union type not handled in Cranelift ExtractValue case
**Solution**:
- Added `original_return_type: Option<Type>` field to SsaBuilder
- Added `with_return_type()` builder method to pass original return type
- Modified `process_typed_terminator` to auto-wrap return values in Result::Ok when function returns error union
- Fixed ExtractValue `ty` fields: discriminant uses `HirType::U32`, ok_value uses `hir_ok_ty`, err_value uses `hir_err_ty`
- Added `HirType::Union` case to Cranelift backend's ExtractValue handler
**Files Modified**:
- `crates/compiler/src/ssa.rs`: Return value wrapping, fixed ExtractValue types
- `crates/compiler/src/lowering.rs`: Pass return type to SsaBuilder
- `crates/compiler/src/cranelift_backend.rs`: Union ExtractValue handling
- `crates/zyn_parser/tests/zig_e2e_jit.rs`: test_zig_jit_try_expression
**Impact**: Try expressions fully working, 23/24 Zig tests passing (95.8%)

### ✅ Switch Expression Implementation (November 18, 2025)
**Problem**: Switch expressions not working - Cranelift only processed entry block, ignoring all match logic
**Root Cause**: When control flow expressions (switch/match/if) were in return position, the Return statement overwrote the entry block's Branch terminator with Return, making other blocks unreachable
**Solution**:
- Added `continuation_block` field to SsaBuilder to track merge/end blocks
- Control flow expressions set continuation_block when creating branching structure
- Return statement handler places Return on continuation block instead of entry block
- Preserves entry block's Branch terminator, allowing full CFG traversal
**Files Modified**:
- `crates/compiler/src/ssa.rs`: Added continuation_block tracking and fixed process_typed_terminator
- `crates/zyn_parser/src/zig.pest`: Switch expression grammar
- `crates/zyn_parser/src/zig_builder.rs`: build_switch_expr and build_switch_pattern functions
- `crates/zyn_parser/tests/zig_e2e_jit.rs`: test_zig_jit_switch_expression
**Impact**: Switch expressions fully working, 22/23 Zig tests passing (95.7%)

### ✅ Array Indexing in Loops Bug Fix (November 18, 2025)
**Problem**: Stack overflow when accessing arrays inside while loops
**Root Cause**: Infinite recursion in `read_variable_recursive` when reading loop-carried array variables. The function would bounce between loop header and back-edge blocks infinitely without creating phi nodes.
**Solution**: Create placeholder phi node and write to block BEFORE recursing to predecessors (breaks the cycle)
**Files Modified**:
- `crates/compiler/src/ssa.rs` (lines 1806-1863): Phi cycle breaking logic
- `crates/compiler/src/typed_cfg.rs`: Pattern check wiring, match arm extraction
- `crates/zyn_parser/src/zig_builder.rs`: Consistent symbol interning
**Impact**: Fixed test_zig_jit_array_in_loop, all loop tests now passing (20/21 Zig tests)

### ✅ Core Compiler Pipeline
- Parser → TypeChecker → HIR → Cranelift JIT
- 98.9% test coverage (20/21 Zig tests passing)
- Full end-to-end compilation

### ✅ Type System
- Generics, traits, lifetimes, dependent types
- Structural and nominal typing
- Multi-paradigm support (OOP, FP, procedural)
- Advanced type features (associated types, GADTs)

### ✅ Async/Await Runtime
- Complete executor, task, waker infrastructure
- Parameter capture in async state machines
- Integration with tiered JIT
- Zero-cost futures

### ✅ Standard Library (Core)
- Vec<T>, String, HashMap<K,V>
- Iterator trait with 50+ adapters
- Basic memory management
- 93/100 stdlib functions compile

### ✅ Tiered JIT Compilation
- 3-tier optimization (Baseline → Standard → Optimized)
- Runtime profiling with atomic counters
- Hot-path detection and recompilation
- Cranelift (fast) + LLVM (optimized) backends

### ✅ HIR Builder API
- Fluent interface for HIR construction
- Type-safe SSA value management
- Zero-cost abstractions

---

## Priority Matrix

| Priority | Category | Item | Effort | Impact |
|----------|----------|------|--------|--------|
| **HIGH** | Ecosystem | Reflaxe/Haxe Integration | 3-4 weeks | Massive |
| **HIGH** | Testing | Fix remaining 4 test failures | 1-2 days | High |
| **MEDIUM** | Compiler | LLVM AOT Backend (CPU) | 2 weeks | High |
| **MEDIUM** | Compiler | GPU kernels (Metal first, then NVPTX); see section 3 | Not estimated | High |
| **MEDIUM** | Language | Exception Handling | 1 week | Medium |
| **MEDIUM** | Stdlib | I/O and File System | 1-2 weeks | High |
| **LOW** | Tooling | LSP Support | 2-3 weeks | Medium |
| **LOW** | Compiler | Bytecode Interpreter | 2 weeks | Low |

---

## Next Milestones

### Milestone 1: Core Stabilization (Q4 2025 - Current)

- ✅ Fix continue statement bug
- ✅ Complete Zig control flow support
- ✅ Array types and indexing
- ✅ Fix array indexing in loops (SSA phi node cycle breaking)
- ✅ String literals support
- ✅ Pattern matching infrastructure (Some/Ok/Err working)
- 🔄 Fix None literal resolution (arena symbol issue)
- 🔄 String operations
- 🔄 Switch expressions and generic functions

### Milestone 2: Production Features (Q1 2026)

- LLVM AOT backend completion (CPU)
- ~~GPU AOT backend Phase 1-2 (NVPTX target, HIR primitives)~~ (not done; superseded 2026-09-25, see section 3)
- Exception handling (try/catch/finally)
- Complete I/O and networking stdlib
- Advanced pattern matching
- Generic functions with type parameters

### Milestone 3: Ecosystem & GPU Integration (Q2 2026)

- Complete Reflaxe/Haxe integration
- ~~GPU AOT backend Phase 3-4 (tensor cores, CUDA runtime, DSL integration)~~ (not done; superseded 2026-09-25, see section 3)
- ~~ZynML `compute()` end-to-end working with GPU~~ (not done; superseded 2026-09-25, see section 3)
- Run 100+ Haxe projects through Zyntax
- Performance benchmarks vs other targets
- 100% test pass rate

### Milestone 4: Developer Experience (Q3 2026)

- LSP implementation
- Package manager
- Comprehensive documentation
- VSCode/IntelliJ integration

---

## Contributing

See individual task items for details. Most tasks have clear acceptance criteria and can be tackled independently.

**Key Resources**:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [docs/HIR_BUILDER_EXAMPLE.md](docs/HIR_BUILDER_EXAMPLE.md) - HIR construction guide
- [docs/ASYNC_RUNTIME_DESIGN.md](docs/ASYNC_RUNTIME_DESIGN.md) - Async runtime internals

**Getting Started**:
1. Pick a task from the backlog
2. Check related documentation
3. Run existing tests to understand the system
4. Implement incrementally with tests
5. Submit PR with clear description

---

## Notes

- Keep this document updated as work progresses
- Move completed items to the Archive section
- Update priority matrix quarterly
- Track time estimates vs actuals for planning
