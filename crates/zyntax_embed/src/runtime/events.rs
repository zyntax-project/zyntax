//! Finding the runtime events a program declares.
//!
//! A walk over a program's statements, collecting what it emits so a
//! host can be handed the set before anything runs.

use super::types::RuntimeEvent;
use std::sync::Arc;
use zyntax_typed_ast::typed_ast::{TypedBlock, TypedProgram, TypedStatement};

const INTERNAL_RENDER_EVENT_ALIAS: &str = "__internal_render_event";
const INTERNAL_STREAM_EVENT_ALIAS: &str = "__internal_stream_event";
const INTERNAL_RENDER_EVENT_SYMBOL: &str = "$Zyntax$render_event";
const INTERNAL_STREAM_EVENT_SYMBOL: &str = "$Zyntax$stream_event";

pub(super) fn capture_runtime_events_from_program(
    program: &mut zyntax_typed_ast::TypedProgram,
    events: &mut Vec<RuntimeEvent>,
    sink: Option<&Arc<dyn Fn(&RuntimeEvent) + Send + Sync>>,
) {
    for decl in &mut program.declarations {
        if let zyntax_typed_ast::TypedDeclaration::Function(function) = &mut decl.node {
            if let Some(body) = &mut function.body {
                capture_runtime_events_from_block(body, events, sink);
            }
        }
    }
}

fn capture_runtime_events_from_block(
    block: &mut zyntax_typed_ast::typed_ast::TypedBlock,
    events: &mut Vec<RuntimeEvent>,
    sink: Option<&Arc<dyn Fn(&RuntimeEvent) + Send + Sync>>,
) {
    for stmt in &mut block.statements {
        capture_runtime_events_from_statement(stmt, events, sink);
    }
}

fn capture_runtime_events_from_statement(
    stmt: &mut zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedStatement>,
    events: &mut Vec<RuntimeEvent>,
    sink: Option<&Arc<dyn Fn(&RuntimeEvent) + Send + Sync>>,
) {
    use zyntax_typed_ast::typed_ast::{TypedExpression, TypedStatement};

    match &mut stmt.node {
        TypedStatement::Expression(expr) => {
            if let TypedExpression::Call(call) = &expr.node {
                if let TypedExpression::Variable(callee) = &call.callee.node {
                    let callee_name = resolve_interned_name(*callee);
                    if is_render_event_name(&callee_name) {
                        let value = call
                            .positional_args
                            .first()
                            .map(expression_summary)
                            .unwrap_or_else(|| "<missing>".to_string());
                        let options = call
                            .named_args
                            .iter()
                            .map(|arg| {
                                (
                                    resolve_interned_name(arg.name),
                                    expression_summary(&arg.value),
                                )
                            })
                            .collect::<Vec<_>>();
                        emit_runtime_event(RuntimeEvent::Render { value, options }, events, sink);
                        *expr = Box::new(make_unit_expression(expr.span));
                        return;
                    }

                    if is_stream_event_name(&callee_name) {
                        let pipeline = call
                            .positional_args
                            .first()
                            .map(expression_summary)
                            .unwrap_or_else(|| "<missing>".to_string());
                        let stage_count = call
                            .positional_args
                            .first()
                            .map(count_nested_call_nodes)
                            .unwrap_or(0);
                        emit_runtime_event(
                            RuntimeEvent::Stream {
                                pipeline,
                                stage_count,
                            },
                            events,
                            sink,
                        );
                        *expr = Box::new(make_unit_expression(expr.span));
                        return;
                    }
                }
            }
        }
        TypedStatement::If(if_stmt) => {
            capture_runtime_events_from_block(&mut if_stmt.then_block, events, sink);
            if let Some(else_block) = &mut if_stmt.else_block {
                capture_runtime_events_from_block(else_block, events, sink);
            }
        }
        TypedStatement::While(while_stmt) => {
            capture_runtime_events_from_block(&mut while_stmt.body, events, sink);
        }
        TypedStatement::Block(block) => {
            capture_runtime_events_from_block(block, events, sink);
        }
        TypedStatement::For(for_stmt) => {
            capture_runtime_events_from_block(&mut for_stmt.body, events, sink);
        }
        TypedStatement::ForCStyle(for_stmt) => {
            if let Some(init) = &mut for_stmt.init {
                capture_runtime_events_from_statement(init, events, sink);
            }
            capture_runtime_events_from_block(&mut for_stmt.body, events, sink);
        }
        TypedStatement::Loop(loop_stmt) => match loop_stmt {
            zyntax_typed_ast::typed_ast::TypedLoop::ForEach { body, .. }
            | zyntax_typed_ast::typed_ast::TypedLoop::ForCStyle { body, .. }
            | zyntax_typed_ast::typed_ast::TypedLoop::While { body, .. }
            | zyntax_typed_ast::typed_ast::TypedLoop::DoWhile { body, .. }
            | zyntax_typed_ast::typed_ast::TypedLoop::Infinite { body } => {
                capture_runtime_events_from_block(body, events, sink);
            }
        },
        TypedStatement::Try(try_stmt) => {
            capture_runtime_events_from_block(&mut try_stmt.body, events, sink);
            for catch in &mut try_stmt.catch_clauses {
                capture_runtime_events_from_block(&mut catch.body, events, sink);
            }
            if let Some(finally) = &mut try_stmt.finally_block {
                capture_runtime_events_from_block(finally, events, sink);
            }
        }
        TypedStatement::Select(select_stmt) => {
            for arm in &mut select_stmt.arms {
                capture_runtime_events_from_block(&mut arm.body, events, sink);
            }
            if let Some(default) = &mut select_stmt.default {
                capture_runtime_events_from_block(default, events, sink);
            }
        }
        _ => {}
    }
}

fn resolve_interned_name(name: zyntax_typed_ast::InternedString) -> String {
    name.resolve_global().unwrap_or_default()
}

fn is_render_event_name(name: &str) -> bool {
    name == INTERNAL_RENDER_EVENT_ALIAS || name == INTERNAL_RENDER_EVENT_SYMBOL
}

fn expression_summary(
    expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
) -> String {
    use zyntax_typed_ast::typed_ast::{TypedExpression, TypedLiteral};

    match &expr.node {
        TypedExpression::Variable(name) => resolve_interned_name(*name),
        TypedExpression::Literal(lit) => match lit {
            TypedLiteral::Integer(v) => v.to_string(),
            TypedLiteral::Float(v) => v.to_string(),
            TypedLiteral::Bool(v) => v.to_string(),
            TypedLiteral::String(s) => s.resolve_global().unwrap_or_default(),
            TypedLiteral::Char(c) => c.to_string(),
            TypedLiteral::Unit => "()".to_string(),
            TypedLiteral::Null => "null".to_string(),
            TypedLiteral::Undefined => "undefined".to_string(),
        },
        TypedExpression::Call(call) => {
            let callee = expression_summary(&call.callee);
            let args = call
                .positional_args
                .iter()
                .map(expression_summary)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", callee, args)
        }
        TypedExpression::Field(field) => {
            let object = expression_summary(&field.object);
            let member = resolve_interned_name(field.field);
            format!("{}.{}", object, member)
        }
        TypedExpression::MethodCall(call) => {
            let receiver = expression_summary(&call.receiver);
            let method = resolve_interned_name(call.method);
            format!("{}.{}(...)", receiver, method)
        }
        TypedExpression::Binary(bin) => {
            let left = expression_summary(&bin.left);
            let right = expression_summary(&bin.right);
            format!("({} {:?} {})", left, bin.op, right)
        }
        _ => format!("{:?}", expr.node),
    }
}

fn count_nested_call_nodes(
    expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
) -> usize {
    use zyntax_typed_ast::typed_ast::TypedExpression;

    match &expr.node {
        TypedExpression::Call(call) => {
            let mut total = 1 + count_nested_call_nodes(&call.callee);
            for arg in &call.positional_args {
                total += count_nested_call_nodes(arg);
            }
            total
        }
        TypedExpression::Field(field) => count_nested_call_nodes(&field.object),
        TypedExpression::MethodCall(call) => {
            let mut total = count_nested_call_nodes(&call.receiver);
            for arg in &call.positional_args {
                total += count_nested_call_nodes(arg);
            }
            total
        }
        TypedExpression::Binary(bin) => {
            count_nested_call_nodes(&bin.left) + count_nested_call_nodes(&bin.right)
        }
        _ => 0,
    }
}

fn emit_runtime_event(
    event: RuntimeEvent,
    events: &mut Vec<RuntimeEvent>,
    sink: Option<&Arc<dyn Fn(&RuntimeEvent) + Send + Sync>>,
) {
    if let Some(s) = sink {
        s(&event);
    }
    events.push(event);
}

fn is_stream_event_name(name: &str) -> bool {
    name == INTERNAL_STREAM_EVENT_ALIAS || name == INTERNAL_STREAM_EVENT_SYMBOL
}

fn make_unit_expression(
    span: zyntax_typed_ast::source::Span,
) -> zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression> {
    zyntax_typed_ast::typed_node(
        zyntax_typed_ast::typed_ast::TypedExpression::Literal(
            zyntax_typed_ast::typed_ast::TypedLiteral::Unit,
        ),
        zyntax_typed_ast::Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
        span,
    )
}
