//! Program debug for Lua lowering.

use super::*;

impl DebugMode {
    pub(super) fn of(scopes: &Scopes) -> Self {
        DebugMode {
            on: scopes.debug,
            locals: scopes.debug_getlocal || scopes.debug_setlocal,
            setlocal: scopes.debug_setlocal,
            rebinds: scopes.debug_rebinds,
        }
    }

    pub(super) fn join(self, other: DebugMode) -> Self {
        DebugMode {
            on: self.on || other.on,
            locals: self.locals || other.locals,
            setlocal: self.setlocal || other.setlocal,
            rebinds: self.rebinds || other.rebinds,
        }
    }

    pub(super) fn info(self) -> Option<DebugInfo> {
        self.on.then(|| DebugInfo {
            locals: self.locals,
            setlocal: self.setlocal,
            ..Default::default()
        })
    }

    /// Kept for the chunks `load` compiles while the program runs.
    pub(super) fn remember(self) {
        let bits = self.on as u8
            | (self.locals as u8) << 1
            | (self.setlocal as u8) << 2
            | (self.rebinds as u8) << 3;
        PROGRAM_DEBUG.store(bits, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn program() -> Self {
        let bits = PROGRAM_DEBUG.load(std::sync::atomic::Ordering::Relaxed);
        DebugMode {
            on: bits & 1 != 0,
            locals: bits & 2 != 0,
            setlocal: bits & 4 != 0,
            rebinds: bits & 8 != 0,
        }
    }
}

/// The call registering a chunk with the debug library: its number,
/// its source and short names, what its lowering recorded, and its
/// upvalue accessor, which is added to its functions.
pub(super) fn debug_register(module: &Module<'_>, (source, short): (&str, &str), span: Span) -> St {
    let accessor = upvalue_accessor(module, span);
    let name = upvalue_accessor_name(&module.tag);
    module.functions.borrow_mut().push(accessor);
    let meta = module
        .debug
        .as_ref()
        .map(|d| d.records.borrow().join("\n"))
        .unwrap_or_default();
    expr_stmt(call(
        "zl_dbg_register",
        vec![
            int_lit(module.chunk_index, span),
            str_lit(source, span),
            str_lit(short, span),
            str_lit(&meta, span),
            call(
                "zl_func_of",
                vec![
                    code_of(&name, span),
                    int_lit(library::debug::UP_ARITY, span),
                ],
                Type::Any,
                span,
            ),
        ],
        prim(PrimitiveType::Unit),
        span,
    ))
}

/// Statements returning a chunk as a function value: its record code,
/// variadic, and its key, kept for the debug library when the program
/// keeps its call stack.
pub(super) fn chunk_value(
    code_name: &str,
    chunk_index: i64,
    debug: bool,
    anys: Type,
    span: Span,
) -> Vec<St> {
    let key = chunk_index << 32;
    let record = call(
        "zb_func_new",
        vec![
            code_of(code_name, span),
            int_lit(VARIADIC_ARITY, span),
            node(
                TypedExpression::Array(vec![call(
                    "zb_box_i64",
                    vec![int_lit(key, span)],
                    Type::Any,
                    span,
                )]),
                anys,
                span,
            ),
        ],
        Type::Any,
        span,
    );
    let name = intern("$chunk");
    let mut out = vec![let_(name, Type::Any, record, span)];
    if debug {
        out.push(expr_stmt(call(
            "zl_dbg_closure",
            vec![int_lit(key, span), var(name, Type::Any, span)],
            prim(PrimitiveType::Unit),
            span,
        )));
    }
    out.push(ret(Some(var(name, Type::Any, span)), span));
    out
}

/// The import every program links the library through.
pub(super) fn library_import() -> TypedNode<TypedDeclaration> {
    TypedNode::new(
        TypedDeclaration::Import(zyntax_typed_ast::typed_ast::TypedImport {
            language: Some(intern("lua")),
            module_path: vec![intern(crate::policy::LIBRARY_MODULE)],
            items: Vec::new(),
            span: Span::new(0, 0),
        }),
        Type::Unknown,
        Span::new(0, 0),
    )
}
