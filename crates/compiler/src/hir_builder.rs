//! # HIR Builder API
//!
//! This module provides a high-level API for constructing HIR (High-level Intermediate Representation)
//! directly, without going through TypedAST. This is useful for:
//!
//! - Building standard library types and functions
//! - Language frontend engineers who want to emit HIR directly
//! - Testing and prototyping compiler features
//! - Compiler intrinsics and built-ins
//!
//! ## Design Philosophy
//!
//! The HIR Builder follows the same pattern as LLVM's IRBuilder and Cranelift's FunctionBuilder:
//! - Stateful builder that tracks current insertion point
//! - Type-safe API that prevents invalid HIR construction
//! - Automatic SSA value management
//! - Uses AstArena for string interning (zero-cost string operations)
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use zyntax_compiler::hir_builder::HirBuilder;
//! use zyntax_typed_ast::AstArena;
//!
//! // Create arena and builder
//! let mut arena = AstArena::new();
//! let mut builder = HirBuilder::new("my_module", &mut arena);
//!
//! // Define a simple add function: fn add(a: i32, b: i32) -> i32
//! let func_id = builder.begin_function("add")
//!     .param("a", builder.i32_type())
//!     .param("b", builder.i32_type())
//!     .returns(builder.i32_type())
//!     .build();
//!
//! // Build function body
//! let entry = builder.create_block("entry");
//! builder.set_insert_point(entry);
//!
//! let a = builder.get_param(0);
//! let b = builder.get_param(1);
//! let result = builder.add(a, b, builder.i32_type());
//! builder.ret(result);
//!
//! // Finish and get the HIR module
//! let hir_module = builder.finish();
//! ```

use crate::hir::*;
use indexmap::IndexMap;
use zyntax_typed_ast::{AstArena, InternedString, Span, TypeId};

/// Main builder for constructing HIR modules
pub struct HirBuilder<'arena> {
    /// The module being built
    module: HirModule,

    /// Arena for string interning
    arena: &'arena mut AstArena,

    /// Current function being built (if any)
    current_function: Option<HirId>,

    /// Current block being built (if any)
    current_block: Option<HirId>,

    /// Parameter IDs for the current function (for get_param)
    current_params: Vec<HirId>,

    /// Type cache for common types
    type_cache: TypeCache,
}

/// Cache of commonly used types
struct TypeCache {
    void: HirType,
    bool_type: HirType,
    i8: HirType,
    i16: HirType,
    i32: HirType,
    i64: HirType,
    u8: HirType,
    u16: HirType,
    u32: HirType,
    u64: HirType,
    f32: HirType,
    f64: HirType,
}

impl TypeCache {
    fn new() -> Self {
        TypeCache {
            void: HirType::Void,
            bool_type: HirType::Bool,
            i8: HirType::I8,
            i16: HirType::I16,
            i32: HirType::I32,
            i64: HirType::I64,
            u8: HirType::U8,
            u16: HirType::U16,
            u32: HirType::U32,
            u64: HirType::U64,
            f32: HirType::F32,
            f64: HirType::F64,
        }
    }
}

impl<'arena> HirBuilder<'arena> {
    /// Creates a new HIR builder for the given module name
    pub fn new(module_name: &str, arena: &'arena mut AstArena) -> Self {
        let name = arena.intern_string(module_name);

        HirBuilder {
            module: HirModule {
                id: HirId::new(),
                name,
                functions: IndexMap::new(),
                globals: IndexMap::new(),
                types: IndexMap::new(),
                imports: Vec::new(),
                exports: Vec::new(),
                version: 0,
                dependencies: std::collections::HashSet::new(),
                effects: IndexMap::new(),
                handlers: IndexMap::new(),
            },
            arena,
            current_function: None,
            current_block: None,
            current_params: Vec::new(),
            type_cache: TypeCache::new(),
        }
    }

    /// Interns a string using the AstArena
    pub fn intern(&mut self, s: &str) -> InternedString {
        self.arena.intern_string(s)
    }

    // ========================================
    // Type Construction
    // ========================================

    /// Returns the void type
    pub fn void_type(&self) -> HirType {
        self.type_cache.void.clone()
    }

    /// Returns the bool type
    pub fn bool_type(&self) -> HirType {
        self.type_cache.bool_type.clone()
    }

    /// Returns i8 type
    pub fn i8_type(&self) -> HirType {
        self.type_cache.i8.clone()
    }

    /// Returns i16 type
    pub fn i16_type(&self) -> HirType {
        self.type_cache.i16.clone()
    }

    /// Returns i32 type
    pub fn i32_type(&self) -> HirType {
        self.type_cache.i32.clone()
    }

    /// Returns i64 type
    pub fn i64_type(&self) -> HirType {
        self.type_cache.i64.clone()
    }

    /// Returns u8 type
    pub fn u8_type(&self) -> HirType {
        self.type_cache.u8.clone()
    }

    /// Returns u16 type
    pub fn u16_type(&self) -> HirType {
        self.type_cache.u16.clone()
    }

    /// Returns u32 type
    pub fn u32_type(&self) -> HirType {
        self.type_cache.u32.clone()
    }

    /// Returns u64 type
    pub fn u64_type(&self) -> HirType {
        self.type_cache.u64.clone()
    }

    /// Returns f32 type
    pub fn f32_type(&self) -> HirType {
        self.type_cache.f32.clone()
    }

    /// Returns f64 type
    pub fn f64_type(&self) -> HirType {
        self.type_cache.f64.clone()
    }

    /// Creates a pointer type
    pub fn ptr_type(&self, pointee: HirType) -> HirType {
        HirType::Ptr(Box::new(pointee))
    }

    /// Creates an array type
    pub fn array_type(&self, element: HirType, size: u64) -> HirType {
        HirType::Array(Box::new(element), size)
    }

    /// Creates a struct type
    pub fn struct_type(&mut self, name: Option<&str>, fields: Vec<HirType>) -> HirType {
        let name_interned = name.map(|n| self.intern(n));

        HirType::Struct(HirStructType {
            name: name_interned,
            fields,
            packed: false,
        })
    }

    /// Creates a union/enum type
    pub fn union_type(&mut self, name: Option<&str>, variants: Vec<HirUnionVariant>) -> HirType {
        let name_interned = name.map(|n| self.intern(n));

        HirType::Union(Box::new(HirUnionType {
            name: name_interned,
            variants,
            discriminant_type: Box::new(HirType::U32), // Default discriminant type
            is_c_union: false,
        }))
    }

    /// Creates a generic type with type arguments
    pub fn generic_type(
        &self,
        base: HirType,
        type_args: Vec<HirType>,
        const_args: Vec<HirConstant>,
    ) -> HirType {
        HirType::Generic {
            base: Box::new(base),
            type_args,
            const_args,
        }
    }

    /// Creates an opaque type parameter
    pub fn type_param(&mut self, name: &str) -> HirType {
        let name_id = self.intern(name);
        HirType::Opaque(name_id)
    }

    // ========================================
    // Function Construction
    // ========================================

    /// Begins building a new function
    pub fn begin_function(&mut self, name: &str) -> FunctionBuilder<'_, 'arena> {
        let name_id = self.intern(name);
        FunctionBuilder::new(self, name_id, Vec::new())
    }

    /// Begins building a new generic function
    pub fn begin_generic_function(
        &mut self,
        name: &str,
        type_params: Vec<&str>,
    ) -> FunctionBuilder<'_, 'arena> {
        let name_id = self.intern(name);
        let type_param_names: Vec<_> = type_params.iter().map(|&n| self.intern(n)).collect();

        FunctionBuilder::new(self, name_id, type_param_names)
    }

    /// Begins building an extern function
    pub fn begin_extern_function(
        &mut self,
        name: &str,
        calling_convention: CallingConvention,
    ) -> FunctionBuilder<'_, 'arena> {
        let name_id = self.intern(name);
        let mut fb = FunctionBuilder::new(self, name_id, Vec::new());
        fb.is_external = true;
        fb.calling_convention = calling_convention;
        fb
    }

    /// Sets the current insertion point to a function
    pub fn set_current_function(&mut self, func_id: HirId) {
        self.current_function = Some(func_id);

        // Extract parameter IDs
        if let Some(func) = self.module.functions.get(&func_id) {
            self.current_params = func.signature.params.iter().map(|p| p.id).collect();
        }
    }

    /// Gets the entry block of the current function
    pub fn entry_block(&self) -> HirId {
        let func_id = self.current_function.expect("No current function");
        self.module.functions[&func_id].entry_block
    }

    /// Gets a parameter by index
    pub fn get_param(&self, index: usize) -> HirId {
        self.current_params[index]
    }

    // ========================================
    // Block Construction
    // ========================================

    /// Creates a new basic block in the current function
    pub fn create_block(&mut self, label: &str) -> HirId {
        let func_id = self.current_function.expect("No current function");
        let block_id = HirId::new();
        let label_id = self.intern(label);

        let block = HirBlock {
            id: block_id,
            label: Some(label_id),
            phis: Vec::new(),
            instructions: Vec::new(),
            terminator: HirTerminator::Unreachable, // Placeholder
            dominance_frontier: std::collections::HashSet::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        };

        let func = self.module.functions.get_mut(&func_id).unwrap();
        func.blocks.insert(block_id, block);

        block_id
    }

    /// Sets the current insertion point to a block
    pub fn set_insert_point(&mut self, block: HirId) {
        self.current_block = Some(block);
    }

    /// Returns the current block
    pub fn current_block_id(&self) -> HirId {
        self.current_block.expect("No current block")
    }

    // ========================================
    // Instruction Building
    // ========================================

    /// Creates an integer constant
    pub fn const_i32(&mut self, value: i32) -> HirId {
        let value_id = HirId::new();
        let ty = self.i32_type();

        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Constant(HirConstant::I32(value)),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Creates a boolean constant
    pub fn const_bool(&mut self, value: bool) -> HirId {
        let value_id = HirId::new();
        let ty = self.bool_type();

        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Constant(HirConstant::Bool(value)),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Creates a u64 constant
    pub fn const_u64(&mut self, value: u64) -> HirId {
        let value_id = HirId::new();
        let ty = self.u64_type();

        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Constant(HirConstant::U64(value)),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Creates a u8 constant
    pub fn const_u8(&mut self, value: u8) -> HirId {
        let value_id = HirId::new();
        let ty = self.u8_type();

        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Constant(HirConstant::U8(value)),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Creates an f64 constant
    pub fn const_f64(&mut self, value: f64) -> HirId {
        let value_id = HirId::new();
        let ty = self.f64_type();

        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Constant(HirConstant::F64(value)),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Creates a global string constant and returns a pointer to it
    pub fn string_constant(&mut self, string_data: &str) -> HirId {
        use crate::hir::{HirConstant, HirGlobal, Linkage, Visibility};

        // Create a unique name for this string global
        let global_id = HirId::new();
        let global_name = self.intern(&format!("str_{:?}", global_id));

        // Create the global with the string data
        let global = HirGlobal {
            id: global_id,
            name: global_name,
            ty: self.ptr_type(self.i8_type()),
            initializer: Some(HirConstant::String(self.intern(string_data))),
            is_const: true,
            is_thread_local: false,
            linkage: Linkage::Private,
            visibility: Visibility::Default,
        };

        // Add to module globals
        self.module.globals.insert(global_id, global);

        // Create a value that references this global
        let value_id = HirId::new();
        let ptr_ty = self.ptr_type(self.i8_type());

        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty: ptr_ty,
                kind: HirValueKind::Global(global_id),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Emits an instruction in the current block
    fn emit(&mut self, inst: HirInstruction) {
        let block_id = self.current_block.unwrap();
        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();
        let block = func.blocks.get_mut(&block_id).unwrap();
        block.instructions.push(inst);
    }

    /// Adds two values
    pub fn add(&mut self, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Binary {
            op: BinaryOp::Add,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Subtracts two values
    pub fn sub(&mut self, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Binary {
            op: BinaryOp::Sub,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Compares two values for equality
    pub fn icmp_eq(&mut self, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Binary {
            op: BinaryOp::Eq,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Loads from a pointer
    pub fn load(&mut self, ptr: HirId, ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Load {
            result,
            ty,
            ptr,
            align: 0, // 0 means use natural alignment
            volatile: false,
        });
        result
    }

    /// Stores to a pointer
    pub fn store(&mut self, value: HirId, ptr: HirId) {
        self.emit(HirInstruction::Store {
            value,
            ptr,
            align: 0, // 0 means use natural alignment
            volatile: false,
        });
    }

    /// Calls a function
    pub fn call(&mut self, callee: HirId, args: Vec<HirId>) -> Option<HirId> {
        let result = Some(HirId::new());
        self.emit(HirInstruction::Call {
            result,
            callee: HirCallable::Function(callee),
            args,
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        result
    }

    /// Calls an intrinsic
    pub fn call_intrinsic(&mut self, intrinsic: Intrinsic, args: Vec<HirId>) -> Option<HirId> {
        let result = Some(HirId::new());
        self.emit(HirInstruction::Call {
            result,
            callee: HirCallable::Intrinsic(intrinsic),
            args,
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        result
    }

    /// Calls an external symbol by name (e.g., "$haxe$trace$int")
    pub fn call_symbol(&mut self, symbol: &str, args: Vec<HirId>) -> Option<HirId> {
        let result = Some(HirId::new());
        self.emit(HirInstruction::Call {
            result,
            callee: HirCallable::Symbol(symbol.to_string()),
            args,
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        result
    }

    /// Panics with the abort intrinsic
    pub fn panic(&mut self) {
        self.call_intrinsic(Intrinsic::Panic, Vec::new());
    }

    /// Gets the size of a type in bytes at runtime
    /// Returns usize (u64)
    pub fn size_of_type(&mut self, ty: HirType) -> HirId {
        let result = Some(HirId::new());
        self.emit(HirInstruction::Call {
            result,
            callee: HirCallable::Intrinsic(Intrinsic::SizeOf),
            args: Vec::new(),
            type_args: vec![ty],
            const_args: Vec::new(),
            is_tail: false,
        });
        result.unwrap()
    }

    /// Gets the alignment of a type in bytes at runtime
    /// Returns usize (u64)
    pub fn align_of_type(&mut self, ty: HirType) -> HirId {
        let result = Some(HirId::new());
        self.emit(HirInstruction::Call {
            result,
            callee: HirCallable::Intrinsic(Intrinsic::AlignOf),
            args: Vec::new(),
            type_args: vec![ty],
            const_args: Vec::new(),
            is_tail: false,
        });
        result.unwrap()
    }

    /// Extracts the discriminant from a union/enum
    pub fn extract_discriminant(&mut self, union_val: HirId) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::GetUnionDiscriminant { result, union_val });
        result
    }

    /// Extracts a value from a union variant
    pub fn extract_union_value(
        &mut self,
        union_val: HirId,
        variant_index: u32,
        ty: HirType,
    ) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::ExtractUnionValue {
            result,
            ty,
            union_val,
            variant_index,
        });
        result
    }

    /// Creates a union value
    pub fn create_union(&mut self, variant_index: u32, value: HirId, union_type: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::CreateUnion {
            result,
            union_ty: union_type,
            variant_index,
            value,
        });
        result
    }

    // ========================================
    // Terminators
    // ========================================

    /// Sets the terminator for the current block
    fn set_terminator(&mut self, terminator: HirTerminator) {
        let block_id = self.current_block.unwrap();
        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();
        let block = func.blocks.get_mut(&block_id).unwrap();
        block.terminator = terminator;
    }

    /// Returns from the function with a value
    pub fn ret(&mut self, value: HirId) {
        self.set_terminator(HirTerminator::Return {
            values: vec![value],
        });
    }

    /// Returns from a void function
    pub fn ret_void(&mut self) {
        self.set_terminator(HirTerminator::Return { values: Vec::new() });
    }

    /// Unconditional branch to a block
    pub fn br(&mut self, target: HirId) {
        self.set_terminator(HirTerminator::Branch { target });
    }

    /// Conditional branch
    pub fn cond_br(&mut self, condition: HirId, then_block: HirId, else_block: HirId) {
        self.set_terminator(HirTerminator::CondBranch {
            condition,
            true_target: then_block,
            false_target: else_block,
        });
    }

    /// Switch on discriminant value
    pub fn switch(&mut self, value: HirId, cases: Vec<(HirConstant, HirId)>, default: HirId) {
        self.set_terminator(HirTerminator::Switch {
            value,
            default,
            cases,
        });
    }

    /// Marks block as unreachable
    pub fn unreachable(&mut self) {
        self.set_terminator(HirTerminator::Unreachable);
    }

    /// Adds a PHI node to the current block
    pub fn add_phi(&mut self, ty: HirType, incoming: Vec<(HirId, HirId)>) -> HirId {
        let result = HirId::new();
        let phi = HirPhi {
            result,
            ty,
            incoming,
        };

        let func_id = self.current_function.expect("No current function");
        let block_id = self.current_block.expect("No current block");

        let func = self.module.functions.get_mut(&func_id).unwrap();
        let block = func.blocks.get_mut(&block_id).unwrap();
        block.phis.push(phi);

        result
    }

    // ========================================
    // Helper Methods
    // ========================================

    /// Gets a function by name (returns HirId for calling)
    pub fn get_function_by_name(&self, name: InternedString) -> HirId {
        self.module
            .functions
            .iter()
            .find(|(_, func)| func.name == name)
            .map(|(id, _)| *id)
            .expect("Function not found")
    }

    /// Creates a function reference value (uses Global kind for function IDs)
    pub fn function_ref(&mut self, func_id: HirId) -> HirId {
        let value_id = HirId::new();
        let current_func_id = self.current_function.unwrap();

        // Get the function type from the referenced function BEFORE getting mutable borrow
        let referenced_func = self.module.functions.get(&func_id).unwrap();
        let func_ty = HirType::Function(Box::new(HirFunctionType {
            params: referenced_func
                .signature
                .params
                .iter()
                .map(|p| p.ty.clone())
                .collect(),
            returns: referenced_func.signature.returns.clone(),
            lifetime_params: referenced_func.signature.lifetime_params.clone(),
            is_variadic: referenced_func.signature.is_variadic,
        }));

        // Now get mutable borrow for current function
        let func = self.module.functions.get_mut(&current_func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty: func_ty,
                kind: HirValueKind::Global(func_id),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Bitcasts a value to a different pointer type
    pub fn bitcast(&mut self, operand: HirId, target_ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Cast {
            op: CastOp::Bitcast,
            result,
            ty: target_ty,
            operand,
        });
        result
    }

    /// Creates a null pointer of the given type
    pub fn null_ptr(&mut self, pointee_ty: HirType) -> HirId {
        let value_id = HirId::new();
        let ptr_ty = self.ptr_type(pointee_ty);
        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty: ptr_ty.clone(),
                kind: HirValueKind::Constant(HirConstant::Null(ptr_ty)),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Creates a unit/void value (represented as empty struct constant)
    pub fn unit_value(&mut self) -> HirId {
        let value_id = HirId::new();
        let void_ty = self.void_type();
        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty: void_ty,
                kind: HirValueKind::Constant(HirConstant::Struct(Vec::new())), // Empty struct = unit
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Creates an undefined/uninitialized value of the given type
    pub fn undef(&mut self, ty: HirType) -> HirId {
        let value_id = HirId::new();
        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Undef,
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Integer comparison (Eq, Ne, Lt, Le, Gt, Ge)
    pub fn icmp(&mut self, op: BinaryOp, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Binary {
            op,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Multiplies two values
    pub fn mul(&mut self, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Binary {
            op: BinaryOp::Mul,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Creates a division instruction: result = lhs / rhs
    pub fn div(&mut self, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::Binary {
            op: BinaryOp::Div,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Creates a struct value from field values
    /// This creates a struct by chaining InsertValue operations
    pub fn create_struct(&mut self, struct_ty: HirType, fields: Vec<HirId>) -> HirId {
        // Start with an undefined/zero value
        let mut aggregate = self.undef_value(struct_ty.clone());

        // Insert each field
        for (index, field_value) in fields.into_iter().enumerate() {
            let insert_result = HirId::new();
            self.emit(HirInstruction::InsertValue {
                result: insert_result,
                ty: struct_ty.clone(),
                aggregate,
                value: field_value,
                indices: vec![index as u32],
            });
            aggregate = insert_result;
        }

        aggregate
    }

    /// Creates an undefined value of the given type
    fn undef_value(&mut self, ty: HirType) -> HirId {
        let value_id = HirId::new();
        let func_id = self.current_function.unwrap();
        let func = self.module.functions.get_mut(&func_id).unwrap();

        func.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Undef,
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        value_id
    }

    /// Extracts a field from a struct value
    pub fn extract_struct_field(
        &mut self,
        struct_val: HirId,
        field_index: u32,
        field_ty: HirType,
    ) -> HirId {
        let result = HirId::new();
        self.emit(HirInstruction::ExtractValue {
            result,
            ty: field_ty,
            aggregate: struct_val,
            indices: vec![field_index],
        });
        result
    }

    /// Pointer arithmetic: ptr + offset (in elements, not bytes)
    pub fn ptr_add(&mut self, ptr: HirId, offset: HirId, ptr_ty: HirType) -> HirId {
        let result = HirId::new();

        self.emit(HirInstruction::GetElementPtr {
            result,
            ty: ptr_ty,
            ptr,
            indices: vec![offset],
        });
        result
    }

    /// Stack allocation
    /// Allocates space on the stack for a value of the given type
    /// Returns a pointer to the allocated space
    pub fn alloca(&mut self, ty: HirType) -> HirId {
        let result = HirId::new();
        let ptr_ty = HirType::Ptr(Box::new(ty.clone()));

        self.emit(HirInstruction::Alloca {
            result,
            ty: ptr_ty,
            count: None,
            align: 0, // 0 means use natural alignment
        });
        result
    }

    /// Get element pointer (for struct field access)
    /// Returns a pointer to a field within a struct
    ///
    /// # Arguments
    /// * `ptr` - Pointer to the struct
    /// * `field_index` - Index of the field (0-based)
    /// * `field_ty` - Type of the field
    pub fn get_element_ptr(&mut self, ptr: HirId, field_index: u32, field_ty: HirType) -> HirId {
        let result = HirId::new();
        let ptr_ty = HirType::Ptr(Box::new(field_ty));
        let index_val = self.const_u64(field_index as u64);

        self.emit(HirInstruction::GetElementPtr {
            result,
            ty: ptr_ty,
            ptr,
            indices: vec![index_val],
        });
        result
    }

    /// Zero extend an integer to a larger type
    ///
    /// # Arguments
    /// * `operand` - The value to extend
    /// * `target_ty` - The target type (must be larger integer type)
    pub fn zext(&mut self, operand: HirId, target_ty: HirType) -> HirId {
        let result = HirId::new();

        self.emit(HirInstruction::Cast {
            op: CastOp::ZExt,
            result,
            ty: target_ty,
            operand,
        });
        result
    }

    /// Bitwise XOR operation
    ///
    /// # Arguments
    /// * `lhs` - Left operand
    /// * `rhs` - Right operand
    /// * `ty` - Result type
    pub fn xor(&mut self, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();

        self.emit(HirInstruction::Binary {
            op: BinaryOp::Xor,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Unsigned remainder (modulo) operation
    ///
    /// # Arguments
    /// * `lhs` - Left operand
    /// * `rhs` - Right operand
    /// * `ty` - Result type
    pub fn urem(&mut self, lhs: HirId, rhs: HirId, ty: HirType) -> HirId {
        let result = HirId::new();

        self.emit(HirInstruction::Binary {
            op: BinaryOp::Rem,
            result,
            ty,
            left: lhs,
            right: rhs,
        });
        result
    }

    /// Check if a function with the given name exists
    ///
    /// # Arguments
    /// * `name` - The interned function name
    pub fn has_function(&self, name: InternedString) -> bool {
        self.module.functions.iter().any(|(_, f)| f.name == name)
    }

    /// Create a function type
    ///
    /// # Arguments
    /// * `params` - Parameter types
    /// * `return_ty` - Return type
    pub fn function_type(&mut self, params: Vec<HirType>, return_ty: HirType) -> HirType {
        HirType::Function(Box::new(HirFunctionType {
            params,
            returns: vec![return_ty],
            lifetime_params: vec![],
            is_variadic: false,
        }))
    }

    // ========================================
    // Finalization
    // ========================================

    /// Finishes building and returns the HIR module
    pub fn finish(self) -> HirModule {
        self.module
    }
}

/// Builder for constructing functions
pub struct FunctionBuilder<'b, 'arena> {
    builder: &'b mut HirBuilder<'arena>,
    name: InternedString,
    params: Vec<HirParam>,
    return_types: Vec<HirType>,
    type_params: Vec<InternedString>,
    is_external: bool,
    calling_convention: CallingConvention,
}

impl<'b, 'arena> FunctionBuilder<'b, 'arena> {
    fn new(
        builder: &'b mut HirBuilder<'arena>,
        name: InternedString,
        type_params: Vec<InternedString>,
    ) -> Self {
        FunctionBuilder {
            builder,
            name,
            params: Vec::new(),
            return_types: Vec::new(),
            type_params,
            is_external: false,
            calling_convention: CallingConvention::Fast,
        }
    }

    /// Adds a parameter to the function
    pub fn param(mut self, name: &str, ty: HirType) -> Self {
        let name_id = self.builder.intern(name);
        let param_id = HirId::new();

        self.params.push(HirParam {
            id: param_id,
            name: name_id,
            ty,
            attributes: ParamAttributes::default(),
        });

        self
    }

    /// Sets the return type
    pub fn returns(mut self, ty: HirType) -> Self {
        self.return_types = vec![ty];
        self
    }

    /// Builds the function and adds it to the module
    pub fn build(self) -> HirId {
        let func_id = HirId::new();

        let type_params: Vec<_> = self
            .type_params
            .into_iter()
            .map(|name| HirTypeParam {
                name,
                constraints: Vec::new(),
            })
            .collect();

        let signature = HirFunctionSignature {
            params: self.params,
            returns: self.return_types,
            type_params,
            const_params: Vec::new(),
            lifetime_params: Vec::new(),
            is_variadic: false,
            is_async: false,
            effects: Vec::new(),
            is_pure: false,
        };

        let mut func = HirFunction::new(self.name, signature.clone());
        func.id = func_id;
        func.is_external = self.is_external;
        func.calling_convention = self.calling_convention;

        // Create Parameter values for each function parameter
        for (index, param) in signature.params.iter().enumerate() {
            let param_value = HirValue {
                id: param.id,
                ty: param.ty.clone(),
                kind: HirValueKind::Parameter(index as u32),
                uses: std::collections::HashSet::new(),
                span: None,
            };
            func.values.insert(param.id, param_value);
        }

        // Clear blocks for extern functions
        if self.is_external {
            func.blocks.clear();
        }

        self.builder.module.functions.insert(func_id, func);
        self.builder.current_function = Some(func_id);

        func_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hir_builder_basic() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_module", &mut arena);

        // Build types first to avoid borrow issues
        let i32_ty = builder.i32_type();

        // Build a simple function: fn add(a: i32, b: i32) -> i32 { return a + b; }
        let func_id = builder
            .begin_function("add")
            .param("a", i32_ty.clone())
            .param("b", i32_ty.clone())
            .returns(i32_ty.clone())
            .build();

        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);

        let a = builder.get_param(0);
        let b = builder.get_param(1);
        let result = builder.add(a, b, i32_ty);
        builder.ret(result);

        let module = builder.finish();

        assert_eq!(module.functions.len(), 1);
        assert!(module.functions.contains_key(&func_id));
    }

    #[test]
    fn test_hir_builder_extern() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_module", &mut arena);

        // Build types first to avoid borrow issues
        let u64_ty = builder.u64_type();
        let u8_ty = builder.u8_type();
        let ptr_u8_ty = builder.ptr_type(u8_ty);

        // Build extern function: extern "C" fn malloc(size: usize) -> *u8
        let _malloc = builder
            .begin_extern_function("malloc", CallingConvention::C)
            .param("size", u64_ty)
            .returns(ptr_u8_ty)
            .build();

        let module = builder.finish();

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.values().next().unwrap();
        assert!(func.is_external);
        assert_eq!(func.calling_convention, CallingConvention::C);
        assert!(func.blocks.is_empty());
    }

    #[test]
    fn test_size_of_intrinsic() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_module", &mut arena);

        // Build types
        let i32_ty = builder.i32_type();
        let f64_ty = builder.f64_type();
        let u64_ty = builder.u64_type();

        // Build function: fn test_sizes() -> usize { return sizeof<i32>() + sizeof<f64>(); }
        let func_id = builder
            .begin_function("test_sizes")
            .returns(u64_ty.clone())
            .build();

        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);

        let i32_size = builder.size_of_type(i32_ty);
        let f64_size = builder.size_of_type(f64_ty);
        let total = builder.add(i32_size, f64_size, u64_ty);
        builder.ret(total);

        let module = builder.finish();

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get(&func_id).unwrap();

        // Count all size_of calls across all blocks
        let mut total_sizeof_calls = 0;
        for (_block_id, block) in &func.blocks {
            for inst in &block.instructions {
                if let HirInstruction::Call { callee, .. } = inst {
                    if matches!(
                        callee,
                        HirCallable::Intrinsic(crate::hir::Intrinsic::SizeOf)
                    ) {
                        total_sizeof_calls += 1;
                    }
                }
            }
        }

        // Should have 2 Call instructions (size_of)
        assert_eq!(
            total_sizeof_calls, 2,
            "Expected 2 SizeOf calls, found {}",
            total_sizeof_calls
        );

        // Verify we also have one Binary add instruction
        let mut binary_count = 0;
        for (_block_id, block) in &func.blocks {
            for inst in &block.instructions {
                if matches!(
                    inst,
                    HirInstruction::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ) {
                    binary_count += 1;
                }
            }
        }
        assert_eq!(
            binary_count, 1,
            "Expected 1 Add instruction, found {}",
            binary_count
        );
    }

    #[test]
    fn test_alloca() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_module", &mut arena);

        let i32_ty = builder.i32_type();
        let void_ty = builder.void_type();

        // Create a function that allocates a local variable
        let func_id = builder
            .begin_function("test_alloca")
            .returns(void_ty)
            .build();

        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);

        // Allocate space for an i32
        let ptr = builder.alloca(i32_ty.clone());

        // Store a value
        let value = builder.const_i32(42);
        builder.store(value, ptr);

        // Load it back
        let _loaded = builder.load(ptr, i32_ty);

        builder.ret_void();

        let module = builder.finish();
        let func = module.functions.get(&func_id).unwrap();

        // Verify we have an Alloca instruction
        let mut alloca_count = 0;
        for (_block_id, block) in &func.blocks {
            for inst in &block.instructions {
                if matches!(inst, HirInstruction::Alloca { .. }) {
                    alloca_count += 1;
                }
            }
        }
        assert_eq!(alloca_count, 1, "Expected 1 Alloca instruction");
    }

    #[test]
    fn test_get_element_ptr() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_module", &mut arena);

        let i32_ty = builder.i32_type();
        let u64_ty = builder.u64_type();

        // Create a struct type: { i32, u64, i32 }
        let struct_ty = builder.struct_type(
            Some("TestStruct"),
            vec![i32_ty.clone(), u64_ty.clone(), i32_ty.clone()],
        );
        let ptr_struct_ty = builder.ptr_type(struct_ty.clone());

        // Create a function that accesses struct fields
        let func_id = builder
            .begin_function("test_gep")
            .param("s", ptr_struct_ty.clone())
            .returns(i32_ty.clone())
            .build();

        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);

        let struct_ptr = builder.get_param(0);

        // Get pointer to field 0 (first i32)
        let field0_ptr = builder.get_element_ptr(struct_ptr, 0, i32_ty.clone());
        let field0_val = builder.load(field0_ptr, i32_ty.clone());

        // Get pointer to field 2 (third field, second i32)
        let field2_ptr = builder.get_element_ptr(struct_ptr, 2, i32_ty.clone());
        let field2_val = builder.load(field2_ptr, i32_ty.clone());

        // Add them
        let result = builder.add(field0_val, field2_val, i32_ty.clone());
        builder.ret(result);

        let module = builder.finish();
        let func = module.functions.get(&func_id).unwrap();

        // Verify we have 2 GetElementPtr instructions
        let mut gep_count = 0;
        for (_block_id, block) in &func.blocks {
            for inst in &block.instructions {
                if matches!(inst, HirInstruction::GetElementPtr { .. }) {
                    gep_count += 1;
                }
            }
        }
        assert_eq!(gep_count, 2, "Expected 2 GetElementPtr instructions");
    }

    #[test]
    fn test_function_type() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_module", &mut arena);

        let i32_ty = builder.i32_type();
        let bool_ty = builder.bool_type();

        // Create a function type: fn(i32, i32) -> bool
        let func_ty = builder.function_type(vec![i32_ty.clone(), i32_ty.clone()], bool_ty.clone());

        // Verify it's a function type
        match func_ty {
            HirType::Function(ref ft) => {
                assert_eq!(ft.params.len(), 2);
                assert_eq!(ft.returns.len(), 1);
                assert_eq!(ft.params[0], i32_ty);
                assert_eq!(ft.params[1], i32_ty);
                assert_eq!(ft.returns[0], bool_ty);
            }
            _ => panic!("Expected function type"),
        }
    }
}
