//! Resolve support.

use super::*;

pub fn resolve(ast: &ast::Ast) -> Scopes {
    resolve_loaded(ast, false)
}

/// [`resolve`] for a chunk `load` compiles: `rebinds` says the program
/// loading it may join or set upvalues, which then holds for the
/// chunk's captures as for the program's.
pub fn resolve_loaded(ast: &ast::Ast, rebinds: bool) -> Scopes {
    let mut scopes = walk(ast, false, rebinds);
    // A module variable is one variable for every function reaching
    // it, which `debug.upvaluejoin` cannot rebind for one closure, and
    // which a frame's spilled locals cannot hold by reference: a chunk
    // that may join upvalues or read or set locals captures its
    // outermost locals as any function's, unless it runs as segments,
    // which share them.
    let reaches = scopes.debug_rebinds || scopes.debug_setlocal || scopes.debug_getlocal;
    if reaches && !scopes.split_chunk && scopes.vars.iter().any(|v| v.is_module_var()) {
        scopes = walk(ast, true, rebinds);
    }
    scopes
}

pub(super) fn walk(ast: &ast::Ast, no_module_vars: bool, rebinds: bool) -> Scopes {
    let mut w = Walker {
        out: Scopes::default(),
        frames: Vec::new(),
        metatable_callees: HashSet::new(),
        debug_calls: HashSet::new(),
        tables_indexed: HashSet::new(),
        debug_aliases: HashSet::new(),
        requiring_debug: false,
        require_calls: HashSet::new(),
        no_module_vars,
    };
    w.out.funcs.push(FuncInfo {
        params: Vec::new(),
        is_vararg: true,
        captures: Vec::new(),
        upvalues: vec![Upvalue::Env],
        line: 0,
        body: (0, 0),
        method: false,
        name: "main".to_string(),
        escapes: false,
        top_level: true,
        returns_function: false,
    });
    w.frames.push(Frame {
        id: CHUNK,
        blocks: vec![HashMap::new()],
        labels: Vec::new(),
    });
    w.stmts(ast.nodes(), true);
    w.frames.pop();
    // A local function capturing itself.
    let self_captures: Vec<VarId> = w
        .out
        .local_functions
        .iter()
        .filter(|(var, func)| w.out.func(**func).captures.contains(var))
        .map(|(var, _)| *var)
        .collect();
    for var in self_captures {
        w.out.vars[var.0 as usize].self_captured = true;
    }
    // A long chunk runs as segments; its outermost locals are module
    // variables, as a captured one already is.
    if ast.nodes().stmts().count() > SEGMENT_STATEMENTS {
        w.out.split_chunk = true;
        for v in &mut w.out.vars {
            if v.outermost && v.func == CHUNK {
                v.captured = true;
            }
        }
    }
    // A variable the debug library may rebind is written where no
    // statement shows it: shared through a cell, never known to hold
    // one function. An upvalue's identity is its cell's address, so a
    // chunk that asks for one gives each captured variable a cell too.
    w.out.debug_rebinds |= rebinds;
    if w.out.debug_rebinds || w.out.debug_setlocal || w.out.upvalue_ids {
        let all = w.out.debug_setlocal;
        for v in &mut w.out.vars {
            if all || v.captured {
                v.assigned = true;
            }
        }
    }
    w.out
}

/// Whether every `return` in a block, outside nested functions, has a
/// function expression as its first value.
pub(super) fn returns_functions(block: &Block) -> bool {
    let first_is_function = |r: &ast::Return| {
        let mut e = r.returns().iter().next();
        while let Some(Expression::Parentheses { expression, .. }) = e {
            e = Some(expression);
        }
        matches!(e, Some(Expression::Function(_)))
    };
    if let Some(ast::LastStmt::Return(r)) = block.last_stmt()
        && !first_is_function(r)
    {
        return false;
    }
    block.stmts().all(|stmt| match stmt {
        Stmt::Do(d) => returns_functions(d.block()),
        Stmt::If(i) => {
            returns_functions(i.block())
                && i.else_if()
                    .is_none_or(|e| e.iter().all(|e| returns_functions(e.block())))
                && i.else_block().is_none_or(returns_functions)
        }
        Stmt::While(w) => returns_functions(w.block()),
        Stmt::Repeat(r) => returns_functions(r.block()),
        Stmt::NumericFor(f) => returns_functions(f.block()),
        Stmt::GenericFor(f) => returns_functions(f.block()),
        _ => true,
    })
}
