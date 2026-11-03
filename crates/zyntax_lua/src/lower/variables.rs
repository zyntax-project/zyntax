//! Variables for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A stripped chunk knows no local's name, calls every upvalue `?`
    /// and cannot tell the environment from another table.
    pub(super) fn describe_name(&self, token: &TokenReference) -> Desc {
        let name = ident(token);
        let stripped = self.m.stripped;
        Some(match self.scopes().binding(token) {
            Some(Binding::Local(_)) if stripped => return None,
            Some(Binding::Upvalue(_)) if stripped => "upvalue '?'".to_string(),
            Some(Binding::Local(_)) => format!("local '{name}'"),
            Some(Binding::Upvalue(_)) => format!("upvalue '{name}'"),
            _ if stripped => format!("field '{name}'"),
            _ => format!("global '{name}'"),
        })
    }

    /// The `_ENV` a free name is a field of, described as the variable
    /// it is at the use: what an index error on it names.
    pub(super) fn env_var_desc(&self, upvalue: bool) -> Desc {
        match (upvalue, self.m.stripped) {
            (true, false) => Some("upvalue '_ENV'".to_string()),
            (true, true) => Some("upvalue '?'".to_string()),
            (false, false) => Some("local '_ENV'".to_string()),
            (false, true) => None,
        }
    }

    pub(super) fn describe(&self, e: &Expression) -> Desc {
        match e {
            Expression::Var(Var::Name(token)) => self.describe_name(token),
            Expression::Var(Var::Expression(v)) => {
                let suffixes: Vec<&Suffix> = v.suffixes().collect();
                self.describe_chain(v.prefix(), &suffixes)
            }
            Expression::Parentheses { expression, .. } => self.describe(expression),
            Expression::String(t) => {
                let bytes = string_bytes(t).ok()?;
                Some(format!("constant '{}'", String::from_utf8_lossy(&bytes)))
            }
            _ => None,
        }
    }

    /// What a prefix with suffixes applied names: the last suffix, or
    /// the prefix itself without any.
    pub(super) fn describe_chain(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Desc {
        let Some((last, init)) = suffixes.split_last() else {
            return match prefix {
                Prefix::Name(token) => self.describe_name(token),
                Prefix::Expression(e) => self.describe(e),
                _ => None,
            };
        };
        // `_ENV.x` is the global `x`.
        if init.is_empty()
            && let Prefix::Name(token) = prefix
            && ident(token) == "_ENV"
            && let Suffix::Index(ast::Index::Dot { name, .. }) = last
            && !self.m.stripped
        {
            return Some(format!("global '{}'", ident(name)));
        }
        Self::describe_suffix(last)
    }

    /// What an index suffix names its result: a field, by the key when
    /// that is a constant string, `integer index` for a small integer
    /// and `?` for any other key, as the reference has it.
    pub(super) fn describe_suffix(s: &Suffix) -> Desc {
        let key = match s {
            Suffix::Index(ast::Index::Dot { name, .. }) => ident(name),
            Suffix::Index(ast::Index::Brackets { expression, .. }) => match expression {
                Expression::String(t) => match string_bytes(t) {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                    Err(_) => "?".to_string(),
                },
                Expression::Number(t) => {
                    match crate::host::parse_numeral(t.token().to_string().trim()) {
                        crate::host::Numeral::Int(0..=255) => "integer index".to_string(),
                        _ => "?".to_string(),
                    }
                }
                _ => "?".to_string(),
            },
            _ => return None,
        };
        Some(format!("field '{key}'"))
    }

    pub(super) fn var_ty(&self, v: VarId) -> Ty {
        self.m.inferred.var(v).settled()
    }

    /// Where a variable of this function or an enclosing one lives,
    /// deciding for one not seen yet.
    pub(super) fn storage_of(&mut self, v: VarId) -> Storage {
        if let Some(s) = self.storage.get(&v) {
            return s.clone();
        }
        let info = self.scopes().var(v);
        let ty = self.var_ty(v);
        let s = if info.is_module_var() {
            let symbol = self.m.module_local_symbol(v);
            self.m.declare_module_var(symbol, ty);
            Storage::Module(symbol, ty)
        } else if info.needs_cell() {
            let symbol = intern(&format!("{}$cell{}", info.name, v.0));
            Storage::Cell(symbol, ty)
        } else {
            let symbol = self.local_symbol(v);
            Storage::Local(symbol, ty)
        };
        self.storage.insert(v, s.clone());
        s
    }

    /// Read a variable.
    pub(super) fn read_var(&mut self, v: VarId, span: Span) -> Val {
        match self.storage_of(v) {
            Storage::Local(name, ty) | Storage::Module(name, ty) => Val {
                node: var(name, self.ir(ty), span),
                ty,
            },
            Storage::Cell(name, ty) => {
                let cell = var(name, self.m.anys(), span);
                let element = index(cell, int_lit(0, span), Type::Any, span);
                let node = self.coerce(
                    Val {
                        node: element,
                        ty: Ty::Any,
                    },
                    ty,
                );
                Val { node, ty }
            }
        }
    }

    /// Store into a variable.
    /// `name = v` where `name` is a field of a local or upvalue `_ENV`.
    pub(super) fn write_field(
        &mut self,
        env: VarId,
        name: &str,
        upvalue: bool,
        value: Val,
        span: Span,
    ) -> St {
        let obj = self.read_var(env, span);
        let key = Val {
            node: str_lit(name, span),
            ty: Ty::Str,
        };
        self.index_write(obj, key, value, self.env_var_desc(upvalue), span)
    }

    pub(super) fn write_var(&mut self, v: VarId, value: Val, span: Span) -> St {
        match self.storage_of(v) {
            Storage::Local(name, ty) | Storage::Module(name, ty) => {
                let value = self.coerce(value, ty);
                assign(var(name, self.ir(ty), span), value, span)
            }
            Storage::Cell(name, ty) => {
                let value = self.coerce(value, ty);
                let value = self.coerce(Val { node: value, ty }, Ty::Any);
                let cell = var(name, self.m.anys(), span);
                assign(index(cell, int_lit(0, span), Type::Any, span), value, span)
            }
        }
    }

    /// Declare a local with a value; its first declaration is a `let`,
    /// a cell's is the cell.
    pub(super) fn declare_var(&mut self, v: VarId, value: Val, span: Span) -> St {
        match self.storage_of(v) {
            Storage::Local(name, ty) => {
                let value = self.coerce(value, ty);
                let_(name, self.ir(ty), value, span)
            }
            Storage::Module(name, ty) => {
                let value = self.coerce(value, ty);
                assign(var(name, self.ir(ty), span), value, span)
            }
            Storage::Cell(name, ty) => {
                let value = self.coerce(value, ty);
                let value = self.coerce(Val { node: value, ty }, Ty::Any);
                let mut pre = Vec::new();
                let cell = self.array_of(vec![value], &mut pre, span);
                let_(name, self.m.anys(), block_value(pre, cell, span), span)
            }
        }
    }

    /// Read a name.
    pub(super) fn read_name(&mut self, token: &TokenReference) -> Result<Val> {
        let span = span_of(token);
        let binding = self
            .scopes()
            .binding(token)
            .cloned()
            .unwrap_or_else(|| Binding::Global(ident(token)));
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => Ok(self.read_var(v, span)),
            Binding::Global(name) => self.read_global(&name, span),
            Binding::Field(v, name, upvalue) => {
                let env = self.read_var(v, span);
                let key = Val {
                    node: str_lit(&name, span),
                    ty: Ty::Str,
                };
                Ok(self.field_read(env, key, &name, self.env_var_desc(upvalue), span))
            }
        }
    }

    pub(super) fn read_global(&mut self, name: &str, span: Span) -> Result<Val> {
        if self.scopes().dynamic_globals {
            // `_ENV` is the environment itself, not an entry in it.
            if name == "_ENV" {
                let g = self.globals_table(span);
                return Ok(match g.ty {
                    Ty::Any => g,
                    _ => Val {
                        node: self.box_table(g.node),
                        ty: Ty::Any,
                    },
                });
            }
            let key = Val {
                node: str_lit(name, span),
                ty: Ty::Str,
            };
            let g = self.globals_table(span);
            let desc = self.m.env_var.and_then(|_| self.env_var_desc(true));
            return Ok(self.index_read(g, key, desc, span));
        }
        if let Some(f) = self.scopes().known_global_function(name) {
            return Ok(self.function_value(f, span));
        }
        if !self.scopes().global_writes.contains_key(name) {
            if let Some(v) = self.preset_value(name, span) {
                return Ok(v);
            }
            if name == "_G" || name == "_ENV" {
                return unsupported("`_G` other than as `_G.name`", span);
            }
            return Ok(self.nil_val(span));
        }
        let ty = self.typer().global_ty(name);
        let symbol = Module::global_symbol(name);
        self.m.declare_module_var(symbol, ty);
        Ok(Val {
            node: var(symbol, self.ir(ty), span),
            ty,
        })
    }

    pub(super) fn write_global(&mut self, name: &str, value: Val, span: Span) -> Result<St> {
        if self.scopes().dynamic_globals {
            // `_ENV = v`: the chunk's environment from here on.
            if name == "_ENV" {
                let Some(env) = self.m.env_var else {
                    return unsupported("assigning `_ENV`", span);
                };
                let v = self.boxed(value);
                return Ok(assign(var(env, Type::Any, span), v, span));
            }
            let key = Val {
                node: str_lit(name, span),
                ty: Ty::Str,
            };
            let g = self.globals_table(span);
            let desc = self.m.env_var.and_then(|_| self.env_var_desc(true));
            return Ok(self.index_write(g, key, value, desc, span));
        }
        if self.scopes().known_global_function(name).is_some() {
            // The one declaration of a known function is its typed
            // entry; nothing is stored.
            return Ok(stmt(
                TypedStatement::Block(TypedBlock {
                    statements: Vec::new(),
                    span,
                }),
                span,
            ));
        }
        let ty = self.typer().global_ty(name);
        let symbol = Module::global_symbol(name);
        self.m.declare_module_var(symbol, ty);
        let value = self.coerce(value, ty);
        Ok(assign(var(symbol, self.ir(ty), span), value, span))
    }

    /// What the global `name` holds when the program starts, when it
    /// holds anything.
    pub(super) fn preset_value(&mut self, name: &str, span: Span) -> Option<Val> {
        if let Some(b) = types::builtin_global(name) {
            return Some(self.builtin_value(b, span));
        }
        let node = if crate::library::stdlib::LIBS.contains(&name) {
            call(
                &crate::library::stdlib::lib_table_fn(name),
                vec![],
                Type::Any,
                span,
            )
        } else if name == "arg" {
            call("zl_arg_table", vec![], Type::Any, span)
        } else if name == "_VERSION" {
            return Some(Val {
                node: str_lit(crate::library::stdlib::LUA_VERSION, span),
                ty: Ty::Str,
            });
        } else {
            return None;
        };
        Some(Val { node, ty: Ty::Any })
    }

    /// Statements giving each global the program assigns, and may read
    /// before it does, the value it holds when the program starts.
    pub(super) fn preset_globals(&mut self, span: Span) -> Vec<St> {
        let scopes = self.scopes();
        if scopes.dynamic_globals {
            return Vec::new();
        }
        let names: Vec<String> = scopes
            .global_writes
            .keys()
            .filter(|name| {
                types::preset_global(name) && !scopes.globals_initialized.contains(*name)
            })
            .cloned()
            .collect();
        let mut out = Vec::new();
        for name in names {
            let Some(value) = self.preset_value(&name, span) else {
                continue;
            };
            let ty = self.typer().global_ty(&name);
            let symbol = Module::global_symbol(&name);
            self.m.declare_module_var(symbol, ty);
            let value = self.coerce(value, ty);
            out.push(assign(var(symbol, self.ir(ty), span), value, span));
        }
        out
    }

    /// The globals table, when the program reaches its globals through
    /// one: every global is an entry, the builtins included. A chunk
    /// with an environment of its own reads it as a value, since
    /// `_ENV = v` may make it anything.
    pub(super) fn globals_table(&mut self, span: Span) -> Val {
        match self.m.env_var {
            Some(env) => Val {
                node: var(env, Type::Any, span),
                ty: Ty::Any,
            },
            None => Val {
                node: var(intern(library::GLOBALS), self.ir(Ty::Table), span),
                ty: Ty::Table,
            },
        }
    }

    /// A builtin as a function value.
    pub(super) fn builtin_value(&mut self, b: &Builtin, span: Span) -> Val {
        let index = crate::library::stdlib::builtin_index(b);
        Val {
            node: call(
                "zl_builtin_value",
                vec![
                    int_lit(index as i64, span),
                    code_of(&crate::library::stdlib::wrapper_name(b), span),
                    int_lit(VARIADIC_ARITY, span),
                ],
                Type::Any,
                span,
            ),
            ty: Ty::Any,
        }
    }

    /// The record of a function: its code, its arity and what it
    /// captures, made where the function is defined or named.
    pub(super) fn function_value(&mut self, f: FuncId, span: Span) -> Val {
        let info = self.scopes().func(f);
        let arity = if info.is_vararg {
            VARIADIC_ARITY
        } else {
            arity_word(info.params.len(), info.params.len())
        };
        let captures = info.captures.clone();
        let mut cells = Vec::with_capacity(captures.len() + 1);
        cells.push(call(
            "zb_box_i64",
            vec![int_lit(self.m.func_key(f), span)],
            Type::Any,
            span,
        ));
        for v in captures {
            cells.push(self.capture_value(v, span));
        }
        let mut pre = Vec::new();
        let cells = self.array_of(cells, &mut pre, span);
        let record = call(
            "zb_func_new",
            vec![
                code_of(&self.m.code_name(f), span),
                int_lit(arity, span),
                cells,
            ],
            Type::Any,
            span,
        );
        // The debug library answers `getinfo(level, "f")` with the
        // last value made of each function.
        let record = if self.m.debug.is_some() {
            let name = self.temp();
            pre.push(let_(name, Type::Any, record, span));
            pre.push(expr_stmt(call(
                "zl_dbg_closure",
                vec![
                    int_lit(self.m.func_key(f), span),
                    var(name, Type::Any, span),
                ],
                prim(PrimitiveType::Unit),
                span,
            )));
            var(name, Type::Any, span)
        } else {
            record
        };
        Val {
            node: block_value(pre, record, span),
            ty: Ty::Any,
        }
    }

    /// What a closure's record holds for a captured variable: the cell
    /// itself, boxed, or the value copied.
    pub(super) fn capture_value(&mut self, v: VarId, span: Span) -> Node {
        match self.storage_of(v) {
            Storage::Cell(name, _) => call(
                "zb_list_box_any",
                vec![var(name, self.m.anys(), span)],
                Type::Any,
                span,
            ),
            Storage::Local(..) | Storage::Module(..) => {
                let value = self.read_var(v, span);
                self.boxed(value)
            }
        }
    }

    /// The prologue of a function taking its record: each capture
    /// read out of `env`.
    pub(super) fn env_prologue(&mut self, span: Span) -> Vec<St> {
        let mut out = Vec::new();
        let captures = self.scopes().func(self.func).captures.clone();
        for (i, v) in captures.iter().enumerate() {
            let slot = index(
                var(intern("env"), self.m.anys(), span),
                int_lit((RECORD_CELLS_AT + i) as i64, span),
                Type::Any,
                span,
            );
            let ty = self.var_ty(*v);
            let info = self.scopes().var(*v);
            if info.needs_cell() {
                let symbol = intern(&format!("{}$cell{}", info.name, v.0));
                self.storage.insert(*v, Storage::Cell(symbol, ty));
                out.push(let_(
                    symbol,
                    self.m.anys(),
                    call("zb_unbox_list_raw_any", vec![slot], self.m.anys(), span),
                    span,
                ));
            } else {
                let symbol = self.local_symbol(*v);
                self.storage.insert(*v, Storage::Local(symbol, ty));
                let value = self.coerce(
                    Val {
                        node: slot,
                        ty: Ty::Any,
                    },
                    ty,
                );
                out.push(let_(symbol, self.ir(ty), value, span));
            }
        }
        out
    }
}
