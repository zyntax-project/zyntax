//! Early errors for declaration files, where every context is ambient.

use oxc_ast::ast as js;
use oxc_span::GetSpan;

use crate::{Error, ErrorKind};

pub(crate) fn is_declaration_file(file: &str) -> bool {
    [".d.ts", ".d.mts", ".d.cts"]
        .iter()
        .any(|suffix| file.ends_with(suffix))
}

pub(crate) fn check(program: &js::Program<'_>) -> Option<Error> {
    statements(&program.body, true)
}

fn error(message: &str, span: oxc_span::Span) -> Option<Error> {
    Some(Error::new(ErrorKind::Type, message, span))
}

fn statements(body: &[js::Statement<'_>], top_level: bool) -> Option<Error> {
    body.iter()
        .find_map(|statement| statement_error(statement, top_level))
}

fn statement_error(statement: &js::Statement<'_>, top_level: bool) -> Option<Error> {
    if let Some(module) = statement.as_module_declaration() {
        return match module {
            js::ModuleDeclaration::ExportDeclaration(export) => {
                declaration_error(&export.declaration)
            }
            js::ModuleDeclaration::ExportDefaultDeclaration(export) => match &export.declaration {
                js::ExportDefaultDeclarationKind::FunctionDeclaration(function) => {
                    function_error(function)
                }
                js::ExportDefaultDeclarationKind::ClassDeclaration(class) => class_error(class),
                _ => None,
            },
            _ => None,
        };
    }
    let Some(declaration) = statement.as_declaration() else {
        return error(
            "statements are not allowed in ambient contexts",
            statement.span(),
        );
    };
    if top_level && !is_ambient_marked(declaration) {
        return error(
            "top-level declarations in .d.ts files must start with either a `declare` or `export` modifier",
            declaration.span(),
        );
    }
    declaration_error(declaration)
}

// Interfaces and type aliases carry no runtime meaning and need no modifier.
fn is_ambient_marked(declaration: &js::Declaration<'_>) -> bool {
    match declaration {
        js::Declaration::VariableDeclaration(d) => d.declare,
        js::Declaration::FunctionDeclaration(d) => d.declare,
        js::Declaration::ClassDeclaration(d) => d.declare,
        js::Declaration::TSEnumDeclaration(d) => d.declare,
        js::Declaration::TSExternalModuleDeclaration(d) => d.declare,
        js::Declaration::TSNamespaceDeclaration(d) => d.declare,
        js::Declaration::TSGlobalDeclaration(d) => d.declare,
        _ => true,
    }
}

fn declaration_error(declaration: &js::Declaration<'_>) -> Option<Error> {
    match declaration {
        js::Declaration::FunctionDeclaration(function) => function_error(function),
        js::Declaration::ClassDeclaration(class) => class_error(class),
        js::Declaration::TSExternalModuleDeclaration(module) => module
            .body
            .as_ref()
            .and_then(|block| statements(&block.body, false)),
        js::Declaration::TSGlobalDeclaration(global) => statements(&global.body.body, false),
        js::Declaration::TSNamespaceDeclaration(namespace) => namespace_error(&namespace.body),
        _ => None,
    }
}

fn namespace_error(body: &js::TSNamespaceDeclarationBody<'_>) -> Option<Error> {
    match body {
        js::TSNamespaceDeclarationBody::TSModuleBlock(block) => statements(&block.body, false),
        js::TSNamespaceDeclarationBody::TSNamespaceDeclaration(inner) => {
            namespace_error(&inner.body)
        }
    }
}

fn function_error(function: &js::Function<'_>) -> Option<Error> {
    if function.generator {
        return error(
            "generators are not allowed in an ambient context",
            function.span,
        );
    }
    function.body.as_ref().and_then(|_| {
        error(
            "an implementation cannot be declared in ambient contexts",
            function.span,
        )
    })
}

fn class_error(class: &js::Class<'_>) -> Option<Error> {
    class.body.body.iter().find_map(|element| match element {
        js::ClassElement::MethodDefinition(method) => function_error(&method.value),
        _ => None,
    })
}
