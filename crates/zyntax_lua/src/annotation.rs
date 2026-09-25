//! LuaLS annotations: the `---@` comments before a statement, which Lua
//! itself skips and the Lua language server reads as types. The
//! frontend registers them with what the statement declares (see
//! [`crate::scope::Declared`]): a function's `@param` and `@return`, a
//! table's `@class` and the `@field`s of its instances, and a stored
//! field's `@type`.

use full_moon::ast::Stmt;
use full_moon::node::Node;
use full_moon::tokenizer::TokenType;

/// A type as an annotation writes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LuaType {
    Nil,
    Boolean,
    Integer,
    Number,
    String,
    Table,
    Function,
    Any,
    /// A class, by the name a `@class` gives it.
    Named(String),
    /// `T?`: the type or nil.
    Optional(Box<LuaType>),
    /// `T|U`.
    Union(Vec<LuaType>),
    /// `T[]`.
    Array(Box<LuaType>),
    /// `fun(a: T): R`.
    Fun {
        params: Vec<LuaType>,
        returns: Vec<LuaType>,
    },
    /// A form the host has no counterpart for: a table shape, a
    /// generic, a literal; as written.
    Other(String),
}

/// One `---@` line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Annotation {
    Class(String),
    Field(String, LuaType),
    Param(String, LuaType),
    Return(Vec<LuaType>),
    Type(LuaType),
}

/// The annotations written before `stmt`, each with its comment's span.
pub(crate) fn before(stmt: &Stmt) -> Vec<((usize, usize), Annotation)> {
    let Some(first) = stmt.tokens().next() else {
        return Vec::new();
    };
    first
        .leading_trivia()
        .filter_map(|t| match t.token_type() {
            TokenType::SingleLineComment { comment } => {
                let span = (t.start_position().bytes(), t.end_position().bytes());
                comment
                    .strip_prefix("-@")
                    .and_then(parse)
                    .map(|a| (span, a))
            }
            _ => None,
        })
        .collect()
}

/// `class Counter`, `param by integer`: an annotation without its `---@`;
/// `None` for a tag the frontend does not read.
fn parse(line: &str) -> Option<Annotation> {
    let (tag, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    let rest = rest.trim();
    Some(match tag {
        "class" => {
            // `@class (exact) Name: Parent`
            let rest = rest.strip_prefix("(exact)").unwrap_or(rest).trim_start();
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
                .collect();
            if name.is_empty() {
                return None;
            }
            Annotation::Class(name)
        }
        "field" => {
            // `@field [public|private|protected|package] name[?] type`
            let rest = ["public ", "private ", "protected ", "package "]
                .iter()
                .find_map(|scope| rest.strip_prefix(scope))
                .unwrap_or(rest)
                .trim_start();
            let (name, optional, ty) = named(rest)?;
            Annotation::Field(name, optional_if(optional, ty))
        }
        "param" => {
            let (name, optional, ty) = named(rest)?;
            Annotation::Param(name, optional_if(optional, ty))
        }
        "return" => {
            // `@return integer count, string name`
            let mut types = Vec::new();
            let mut at = rest;
            loop {
                let (ty, after) = Reader::new(at).whole()?;
                types.push(ty);
                // An optional name, then a comma for another.
                let after = after.trim_start();
                let after = after
                    .strip_prefix(|c: char| c.is_alphabetic() || c == '_')
                    .map(|_| after.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_'))
                    .unwrap_or(after)
                    .trim_start();
                match after.strip_prefix(',') {
                    Some(next) => at = next.trim_start(),
                    None => break,
                }
            }
            Annotation::Return(types)
        }
        "type" => Annotation::Type(Reader::new(rest).whole()?.0),
        _ => return None,
    })
}

/// `name[?] type`: the name, whether it is optional, and the type.
fn named(text: &str) -> Option<(String, bool, LuaType)> {
    let (name, rest) = text.split_once(char::is_whitespace)?;
    let (name, optional) = match name.strip_suffix('?') {
        Some(name) => (name, true),
        None => (name, false),
    };
    Some((
        name.to_owned(),
        optional,
        Reader::new(rest.trim()).whole()?.0,
    ))
}

fn optional_if(optional: bool, ty: LuaType) -> LuaType {
    if optional {
        LuaType::Optional(Box::new(ty))
    } else {
        ty
    }
}

/// A reader of a type's text.
struct Reader<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(text: &'a str) -> Reader<'a> {
        Reader { text, at: 0 }
    }

    /// The type at the start, and the text after it (a name or a
    /// description).
    fn whole(mut self) -> Option<(LuaType, &'a str)> {
        let ty = self.union()?;
        Some((ty, &self.text[self.at..]))
    }

    fn rest(&self) -> &'a str {
        &self.text[self.at..]
    }

    fn skip_space(&mut self) {
        let rest = self.rest();
        self.at += rest.len() - rest.trim_start().len();
    }

    fn eat(&mut self, s: &str) -> bool {
        self.skip_space();
        if self.rest().starts_with(s) {
            self.at += s.len();
            true
        } else {
            false
        }
    }

    fn union(&mut self) -> Option<LuaType> {
        let mut types = vec![self.postfix()?];
        while self.eat("|") {
            types.push(self.postfix()?);
        }
        Some(if types.len() == 1 {
            types.pop().unwrap()
        } else {
            LuaType::Union(types)
        })
    }

    fn postfix(&mut self) -> Option<LuaType> {
        let mut ty = self.primary()?;
        loop {
            if self.rest().starts_with("[]") {
                self.at += 2;
                ty = LuaType::Array(Box::new(ty));
            } else if self.rest().starts_with('?') {
                self.at += 1;
                ty = LuaType::Optional(Box::new(ty));
            } else {
                return Some(ty);
            }
        }
    }

    fn primary(&mut self) -> Option<LuaType> {
        self.skip_space();
        let rest = self.rest();
        if self.eat("(") {
            let ty = self.union()?;
            return self.eat(")").then_some(ty);
        }
        if rest.starts_with('{') || rest.starts_with('"') || rest.starts_with('\'') {
            let len = balanced(rest)?;
            self.at += len;
            return Some(LuaType::Other(rest[..len].to_owned()));
        }
        let name_len = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '-'))
            .unwrap_or(rest.len());
        if name_len == 0 {
            return None;
        }
        let name = &rest[..name_len];
        self.at += name_len;
        if name == "fun" && self.rest().starts_with('(') {
            return self.fun();
        }
        if self.rest().starts_with('<') {
            // `table<K, V>` and other generics.
            let len = balanced(self.rest())?;
            let written = format!("{name}{}", &self.rest()[..len]);
            self.at += len;
            return Some(LuaType::Other(written));
        }
        Some(match name {
            "nil" | "void" => LuaType::Nil,
            "boolean" | "bool" => LuaType::Boolean,
            "integer" => LuaType::Integer,
            "number" => LuaType::Number,
            "string" => LuaType::String,
            "table" => LuaType::Table,
            "function" => LuaType::Function,
            "any" | "unknown" => LuaType::Any,
            _ if name.starts_with(|c: char| c.is_ascii_digit() || c == '-') => {
                LuaType::Other(name.to_owned())
            }
            _ => LuaType::Named(name.to_owned()),
        })
    }

    /// `fun(a: T, b?: U): R, S`, after `fun`.
    fn fun(&mut self) -> Option<LuaType> {
        self.eat("(");
        let mut params = Vec::new();
        if !self.eat(")") {
            loop {
                self.skip_space();
                // `name: T`, `name?: T`, `...: T`, or an untyped `name`.
                let rest = self.rest();
                let end = rest.find([':', ',', ')'])?;
                let optional = rest[..end].trim_end().ends_with('?');
                self.at += end;
                let ty = if self.eat(":") {
                    self.union()?
                } else {
                    LuaType::Any
                };
                params.push(optional_if(optional, ty));
                if self.eat(")") {
                    break;
                }
                if !self.eat(",") {
                    return None;
                }
            }
        }
        let mut returns = Vec::new();
        if self.eat(":") {
            returns.push(self.union()?);
            while self.eat(",") {
                returns.push(self.union()?);
            }
        }
        Some(LuaType::Fun { params, returns })
    }
}

/// The length of the bracketed or quoted form at the start of `text`.
fn balanced(text: &str) -> Option<usize> {
    let open = text.chars().next()?;
    if open == '"' || open == '\'' {
        return text[1..].find(open).map(|i| i + 2);
    }
    let close = match open {
        '{' => '}',
        '<' => '>',
        '(' => ')',
        _ => return None,
    };
    let mut depth = 0;
    for (i, c) in text.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(line: &str) -> Annotation {
        parse(line).unwrap_or_else(|| panic!("{line}"))
    }

    #[test]
    fn tags_read_as_luals_writes_them() {
        assert_eq!(one("class Counter"), Annotation::Class("Counter".into()));
        assert_eq!(
            one("class (exact) game.Counter: Base"),
            Annotation::Class("game.Counter".into())
        );
        assert_eq!(
            one("field n integer the count"),
            Annotation::Field("n".into(), LuaType::Integer)
        );
        assert_eq!(
            one("field private label? string"),
            Annotation::Field("label".into(), LuaType::Optional(Box::new(LuaType::String)))
        );
        assert_eq!(
            one("param by integer how far"),
            Annotation::Param("by".into(), LuaType::Integer)
        );
        assert_eq!(
            one("return integer count, string name"),
            Annotation::Return(vec![LuaType::Integer, LuaType::String])
        );
        assert_eq!(one("type number"), Annotation::Type(LuaType::Number));
        assert_eq!(parse("diagnostic disable"), None);
    }

    #[test]
    fn types_compose() {
        let ty = |text: &str| Reader::new(text).whole().unwrap().0;
        assert_eq!(
            ty("string|nil"),
            LuaType::Union(vec![LuaType::String, LuaType::Nil])
        );
        assert_eq!(
            ty("Counter[]"),
            LuaType::Array(Box::new(LuaType::Named("Counter".into())))
        );
        assert_eq!(
            ty("fun(i: integer, s?: string): number"),
            LuaType::Fun {
                params: vec![
                    LuaType::Integer,
                    LuaType::Optional(Box::new(LuaType::String))
                ],
                returns: vec![LuaType::Number],
            }
        );
        assert_eq!(
            ty("table<string, integer>"),
            LuaType::Other("table<string, integer>".into())
        );
        assert_eq!(ty("{ x: number }"), LuaType::Other("{ x: number }".into()));
        assert_eq!(
            ty("(integer|string)[]"),
            LuaType::Array(Box::new(LuaType::Union(vec![
                LuaType::Integer,
                LuaType::String
            ])))
        );
    }
}
