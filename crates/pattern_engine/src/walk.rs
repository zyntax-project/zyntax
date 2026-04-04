//! Depth-first mutable walker for TypedProgram.
//!
//! Walks bottom-up: children are visited before their parent, so inner rewrites
//! fire first and outer rewrites see already-transformed subtrees.

use crate::context::MatchContext;
use crate::rewrite::{DeclRewrite, ExprRewrite, RewriteOutput, StmtRewrite};
use crate::trace::{FiredRewrite, FiredSet};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::TypedASTBuilder;

/// Result of walking the program in one iteration.
pub struct WalkResult {
    pub changed: bool,
    pub fired: Vec<FiredRewrite>,
}

/// Walk the entire program, applying rewrites at each node.
/// Returns whether any rewrite fired.
pub fn walk_program(
    program: &mut TypedProgram,
    expr_rewrites: &[ExprRewrite],
    stmt_rewrites: &[StmtRewrite],
    decl_rewrites: &[DeclRewrite],
    ctx: &MatchContext,
    builder: &mut TypedASTBuilder,
    iteration: usize,
) -> WalkResult {
    let mut result = WalkResult {
        changed: false,
        fired: Vec::new(),
    };
    let mut fired_set = FiredSet::new();

    // Collect expanded declarations to append after the walk
    let mut pending_decls: Vec<TypedNode<TypedDeclaration>> = Vec::new();

    // Walk each declaration, tracking indices to delete
    let mut delete_indices: Vec<usize> = Vec::new();

    for (idx, decl) in program.declarations.iter_mut().enumerate() {
        // Walk into the declaration's children first (bottom-up)
        walk_declaration_children(
            decl,
            expr_rewrites,
            stmt_rewrites,
            ctx,
            builder,
            iteration,
            &mut result,
            &mut fired_set,
        );

        // Then try decl-level rewrites on this declaration
        for rw in decl_rewrites {
            if let Some(bindings) = rw.pattern.try_match(decl, ctx) {
                let output = (rw.apply)(decl, bindings, builder);
                match output {
                    RewriteOutput::ReplaceDecl(new_decl) => {
                        *decl = new_decl;
                        result.changed = true;
                    }
                    RewriteOutput::Expand {
                        declarations,
                        replacement,
                    } => {
                        if let Some(repl) = replacement {
                            *decl = repl;
                        } else {
                            delete_indices.push(idx);
                        }
                        pending_decls.extend(declarations);
                        result.changed = true;
                    }
                    RewriteOutput::Delete => {
                        delete_indices.push(idx);
                        result.changed = true;
                    }
                    RewriteOutput::Unchanged => {}
                    // Expression/statement replacements don't apply at decl level
                    RewriteOutput::ReplaceExpr(_) | RewriteOutput::ReplaceStmt(_) => {}
                }

                if result.changed {
                    result.fired.push(FiredRewrite {
                        rewrite_id: rw.id,
                        rewrite_name: rw.name,
                        node_span: decl.span,
                        iteration,
                    });
                    fired_set.insert(rw.id);
                    break; // first-match-wins per node
                }
            }
        }
    }

    // Remove deleted declarations (in reverse order to preserve indices)
    for idx in delete_indices.into_iter().rev() {
        program.declarations.remove(idx);
    }

    // Append expanded declarations
    program.declarations.extend(pending_decls);

    result
}

/// Walk into a declaration's children (function bodies, class methods, etc.)
fn walk_declaration_children(
    decl: &mut TypedNode<TypedDeclaration>,
    expr_rewrites: &[ExprRewrite],
    stmt_rewrites: &[StmtRewrite],
    ctx: &MatchContext,
    builder: &mut TypedASTBuilder,
    iteration: usize,
    result: &mut WalkResult,
    fired_set: &mut FiredSet,
) {
    match &mut decl.node {
        TypedDeclaration::Function(func) => {
            if let Some(body) = &mut func.body {
                walk_block(
                    body,
                    expr_rewrites,
                    stmt_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedDeclaration::Class(class) => {
            for method in &mut class.methods {
                if let Some(body) = &mut method.body {
                    walk_block(
                        body,
                        expr_rewrites,
                        stmt_rewrites,
                        ctx,
                        builder,
                        iteration,
                        result,
                        fired_set,
                    );
                }
            }
        }
        TypedDeclaration::Impl(impl_block) => {
            for method in &mut impl_block.methods {
                if let Some(body) = &mut method.body {
                    walk_block(
                        body,
                        expr_rewrites,
                        stmt_rewrites,
                        ctx,
                        builder,
                        iteration,
                        result,
                        fired_set,
                    );
                }
            }
        }
        TypedDeclaration::EffectHandler(handler) => {
            for handler_impl in &mut handler.handlers {
                if let Some(body) = &mut handler_impl.body {
                    walk_block(
                        body,
                        expr_rewrites,
                        stmt_rewrites,
                        ctx,
                        builder,
                        iteration,
                        result,
                        fired_set,
                    );
                }
            }
        }
        _ => {}
    }
}

/// Walk a block of statements.
fn walk_block(
    block: &mut TypedBlock,
    expr_rewrites: &[ExprRewrite],
    stmt_rewrites: &[StmtRewrite],
    ctx: &MatchContext,
    builder: &mut TypedASTBuilder,
    iteration: usize,
    result: &mut WalkResult,
    fired_set: &mut FiredSet,
) {
    let mut delete_indices: Vec<usize> = Vec::new();

    for (idx, stmt) in block.statements.iter_mut().enumerate() {
        // Walk into statement's children first (bottom-up)
        walk_statement_children(
            stmt,
            expr_rewrites,
            ctx,
            builder,
            iteration,
            result,
            fired_set,
        );

        // Then try stmt-level rewrites
        for rw in stmt_rewrites {
            if let Some(bindings) = rw.pattern.try_match(stmt, ctx) {
                let output = (rw.apply)(stmt, bindings, builder);
                let mut did_fire = false;
                match output {
                    RewriteOutput::ReplaceStmt(new_stmt) => {
                        *stmt = new_stmt;
                        did_fire = true;
                    }
                    RewriteOutput::Delete => {
                        delete_indices.push(idx);
                        did_fire = true;
                    }
                    RewriteOutput::Unchanged => {}
                    _ => {}
                }

                if did_fire {
                    result.changed = true;
                    result.fired.push(FiredRewrite {
                        rewrite_id: rw.id,
                        rewrite_name: rw.name,
                        node_span: stmt.span,
                        iteration,
                    });
                    fired_set.insert(rw.id);
                    break; // first-match-wins
                }
            }
        }
    }

    for idx in delete_indices.into_iter().rev() {
        block.statements.remove(idx);
    }
}

/// Walk into a statement's child expressions.
fn walk_statement_children(
    stmt: &mut TypedNode<TypedStatement>,
    expr_rewrites: &[ExprRewrite],
    ctx: &MatchContext,
    builder: &mut TypedASTBuilder,
    iteration: usize,
    result: &mut WalkResult,
    fired_set: &mut FiredSet,
) {
    match &mut stmt.node {
        TypedStatement::Expression(expr) => {
            walk_expression(
                expr,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedStatement::Let(let_stmt) => {
            if let Some(init) = &mut let_stmt.initializer {
                walk_expression(
                    init,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedStatement::Return(Some(expr)) => {
            walk_expression(
                expr,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedStatement::Yield(expr) => {
            walk_expression(
                expr,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedStatement::If(if_stmt) => {
            walk_expression(
                &mut if_stmt.condition,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            walk_block(
                &mut if_stmt.then_block,
                expr_rewrites,
                &[], // no stmt rewrites in nested blocks (they fire at the block walk level)
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            if let Some(else_block) = &mut if_stmt.else_block {
                walk_block(
                    else_block,
                    expr_rewrites,
                    &[],
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedStatement::While(while_stmt) => {
            walk_expression(
                &mut while_stmt.condition,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            walk_block(
                &mut while_stmt.body,
                expr_rewrites,
                &[],
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedStatement::For(for_stmt) => {
            walk_expression(
                &mut for_stmt.iterator,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            walk_block(
                &mut for_stmt.body,
                expr_rewrites,
                &[],
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedStatement::Block(block) => {
            walk_block(
                block,
                expr_rewrites,
                &[],
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedStatement::Match(match_stmt) => {
            walk_expression(
                &mut match_stmt.scrutinee,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            for arm in &mut match_stmt.arms {
                if let Some(guard) = &mut arm.guard {
                    walk_expression(
                        guard,
                        expr_rewrites,
                        ctx,
                        builder,
                        iteration,
                        result,
                        fired_set,
                    );
                }
                walk_expression(
                    &mut arm.body,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedStatement::Throw(expr) => {
            walk_expression(
                expr,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedStatement::Break(Some(expr)) => {
            walk_expression(
                expr,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        _ => {}
    }
}

/// Walk an expression and try expression-level rewrites.
fn walk_expression(
    expr: &mut TypedNode<TypedExpression>,
    expr_rewrites: &[ExprRewrite],
    ctx: &MatchContext,
    builder: &mut TypedASTBuilder,
    iteration: usize,
    result: &mut WalkResult,
    fired_set: &mut FiredSet,
) {
    // Walk children first (bottom-up)
    walk_expression_children(
        expr,
        expr_rewrites,
        ctx,
        builder,
        iteration,
        result,
        fired_set,
    );

    // Then try expr-level rewrites on this node
    for rw in expr_rewrites {
        if let Some(bindings) = rw.pattern.try_match(expr, ctx) {
            let output = (rw.apply)(expr, bindings, builder);
            match output {
                RewriteOutput::ReplaceExpr(new_expr) => {
                    *expr = new_expr;
                    result.changed = true;
                    result.fired.push(FiredRewrite {
                        rewrite_id: rw.id,
                        rewrite_name: rw.name,
                        node_span: expr.span,
                        iteration,
                    });
                    fired_set.insert(rw.id);
                    break; // first-match-wins
                }
                RewriteOutput::Unchanged => {}
                _ => {} // Delete/Expand don't apply at expression level
            }
        }
    }
}

/// Walk into an expression's sub-expressions.
fn walk_expression_children(
    expr: &mut TypedNode<TypedExpression>,
    expr_rewrites: &[ExprRewrite],
    ctx: &MatchContext,
    builder: &mut TypedASTBuilder,
    iteration: usize,
    result: &mut WalkResult,
    fired_set: &mut FiredSet,
) {
    match &mut expr.node {
        TypedExpression::Binary(bin) => {
            walk_expression(
                &mut bin.left,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            walk_expression(
                &mut bin.right,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Unary(un) => {
            walk_expression(
                &mut un.operand,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Call(call) => {
            walk_expression(
                &mut call.callee,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            for arg in &mut call.positional_args {
                walk_expression(
                    arg,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
            for named_arg in &mut call.named_args {
                walk_expression(
                    &mut named_arg.value,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedExpression::Field(field) => {
            walk_expression(
                &mut field.object,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Index(idx) => {
            walk_expression(
                &mut idx.object,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            walk_expression(
                &mut idx.index,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Array(elems) => {
            for elem in elems {
                walk_expression(
                    elem,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedExpression::Tuple(elems) => {
            for elem in elems {
                walk_expression(
                    elem,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedExpression::If(if_expr) => {
            walk_expression(
                &mut if_expr.condition,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            walk_expression(
                &mut if_expr.then_branch,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            walk_expression(
                &mut if_expr.else_branch,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Await(inner) => {
            walk_expression(
                inner,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Try(inner) => {
            walk_expression(
                inner,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Dereference(inner) => {
            walk_expression(
                inner,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Reference(reference) => {
            walk_expression(
                &mut reference.expr,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Cast(cast) => {
            walk_expression(
                &mut cast.expr,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::MethodCall(method_call) => {
            walk_expression(
                &mut method_call.receiver,
                expr_rewrites,
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
            for arg in &mut method_call.positional_args {
                walk_expression(
                    arg,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
            for named_arg in &mut method_call.named_args {
                walk_expression(
                    &mut named_arg.value,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedExpression::Block(block) => {
            walk_block(
                block,
                expr_rewrites,
                &[],
                ctx,
                builder,
                iteration,
                result,
                fired_set,
            );
        }
        TypedExpression::Lambda(lambda) => match &mut lambda.body {
            TypedLambdaBody::Expression(expr) => {
                walk_expression(
                    expr,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
            TypedLambdaBody::Block(block) => {
                walk_block(
                    block,
                    expr_rewrites,
                    &[],
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        },
        TypedExpression::Struct(struct_lit) => {
            for field in &mut struct_lit.fields {
                walk_expression(
                    &mut field.value,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        TypedExpression::Range(range) => {
            if let Some(start) = &mut range.start {
                walk_expression(
                    start,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
            if let Some(end) = &mut range.end {
                walk_expression(
                    end,
                    expr_rewrites,
                    ctx,
                    builder,
                    iteration,
                    result,
                    fired_set,
                );
            }
        }
        // Leaf expressions — nothing to descend into
        TypedExpression::Literal(_) | TypedExpression::Variable(_) | TypedExpression::Path(_) => {}
        // Remaining variants — skip for now, extend as needed
        _ => {}
    }
}
