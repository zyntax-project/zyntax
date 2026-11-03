//! Lua's own reading of a chunk: the lexer and parser of the reference
//! without its code generator, so a chunk the reference refuses is
//! refused here with the reference's message and line, and the chunk
//! `full_moon` is then given is one it can take.
//!
//! A message is worded as `luaX_syntaxerror` words it: `<line>: <what>
//! near <token>`, the token as the reference shows it (`<eof>`, `'='`,
//! or the text of a name, string or numeral as read). Semantic errors
//! (labels, gotos, attributes, read-only variables) carry no token.

mod bindings;
mod expressions;
mod lex;
mod source;
mod statements;

use expressions::*;
use lex::*;
pub(crate) use source::*;

/// The reference's limits, as `luaconf.h` and `lparser.c` set them.
const MAX_C_CALLS: usize = 200;
const MAX_VARS: usize = 200;
const MAX_UPVALUES: usize = 255;
const MAX_REGS: usize = 255;
const FIELDS_PER_FLUSH: usize = 50;

/// Why a chunk does not parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxError {
    /// The line the reference reports, `None` for a message it gives
    /// without a position (a C stack overflow).
    pub line: Option<usize>,
    /// The message after the position.
    pub message: String,
    /// Where in the source the reader stood.
    pub offset: usize,
}

impl SyntaxError {
    /// The message with its position, as `chunk:line: message`.
    pub fn with_chunk(&self, chunk: &str) -> String {
        match self.line {
            Some(line) => format!("{chunk}:{line}: {}", self.message),
            None => self.message.clone(),
        }
    }
}

/// What reading a chunk found that `full_moon` needs told.
#[derive(Debug, Default)]
pub(crate) struct Checked {
    /// Offsets of `break` statements that other statements follow in
    /// their block. Lua 5.4 allows that; `full_moon` takes a `break`
    /// only as the last statement, so each is read as `do break end`.
    pub breaks: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum T {
    Char(u8),
    And,
    Break,
    Do,
    Else,
    Elseif,
    End,
    False,
    For,
    Function,
    Goto,
    If,
    In,
    Local,
    Nil,
    Not,
    Or,
    Repeat,
    Return,
    Then,
    True,
    Until,
    While,
    IDiv,
    Concat,
    Dots,
    Eq,
    Ge,
    Le,
    Ne,
    Shl,
    Shr,
    DbColon,
    Eos,
    Flt,
    Int,
    Name,
    Str,
}

const RESERVED: [(&str, T); 22] = [
    ("and", T::And),
    ("break", T::Break),
    ("do", T::Do),
    ("else", T::Else),
    ("elseif", T::Elseif),
    ("end", T::End),
    ("false", T::False),
    ("for", T::For),
    ("function", T::Function),
    ("goto", T::Goto),
    ("if", T::If),
    ("in", T::In),
    ("local", T::Local),
    ("nil", T::Nil),
    ("not", T::Not),
    ("or", T::Or),
    ("repeat", T::Repeat),
    ("return", T::Return),
    ("then", T::Then),
    ("true", T::True),
    ("until", T::Until),
    ("while", T::While),
];

#[derive(Clone, Copy)]
struct Token<'a> {
    t: T,
    /// A name's text.
    name: &'a [u8],
    /// Where the token starts.
    at: usize,
}

const NO_TOKEN: Token<'static> = Token {
    t: T::Eos,
    name: b"",
    at: 0,
};

/// `llex.c`: tokens on demand, with the line the reader has reached and
/// the text of the token read last, which a message quotes.
struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
    buf: Vec<u8>,
    tok: Token<'a>,
    ahead: Option<Token<'a>>,
}

type R<T> = Result<T, SyntaxError>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Regular,
    Const,
    Close,
}

struct Var<'a> {
    name: &'a [u8],
    kind: Kind,
}

#[derive(Clone, Copy)]
struct Label<'a> {
    name: &'a [u8],
    line: usize,
    nactvar: usize,
}

struct Block {
    first_label: usize,
    first_goto: usize,
    nactvar: usize,
    is_loop: bool,
}

struct Func<'a> {
    first_local: usize,
    first_label: usize,
    nactvar: usize,
    upvalues: Vec<(&'a [u8], Kind)>,
    blocks: Vec<Block>,
    vararg: bool,
    line_defined: usize,
    free_reg: usize,
}

/// What an expression is, as far as a statement needs to know: whether
/// it can be assigned, and whether it is a call.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ek {
    Void,
    Call,
    Vararg,
    Local(Kind),
    Upvalue(Kind),
    Indexed,
    Value,
}

#[derive(Clone, Copy)]
struct Exp<'a> {
    k: Ek,
    name: &'a [u8],
}

impl<'a> Exp<'a> {
    fn of(k: Ek) -> Self {
        Exp { k, name: b"" }
    }
    fn is_var(&self) -> bool {
        matches!(self.k, Ek::Local(_) | Ek::Upvalue(_) | Ek::Indexed)
    }
    fn multiple(&self) -> bool {
        matches!(self.k, Ek::Call | Ek::Vararg)
    }
}

const BREAK: &[u8] = b"break";
const ENV: &[u8] = b"_ENV";

/// `lparser.c` without code: scopes, labels, gotos, attributes and
/// the limits, over the tokens.
struct Parser<'a> {
    lx: Lexer<'a>,
    actvar: Vec<Var<'a>>,
    labels: Vec<Label<'a>>,
    gotos: Vec<Label<'a>>,
    funcs: Vec<Func<'a>>,
    level: usize,
    checked: Checked,
}

const UNARY_PRIORITY: u8 = 12;

#[cfg(test)]
mod tests {
    use super::*;

    fn err(src: &str) -> String {
        check(src.as_bytes(), 2)
            .map(|_| "ok".to_string())
            .unwrap_or_else(|e| e.with_chunk("c"))
    }

    #[test]
    fn messages_match_the_reference() {
        assert_eq!(err("x = "), "c:1: unexpected symbol near <eof>");
        assert_eq!(
            err("local a = {4\n\n"),
            "c:3: '}' expected (to close '{' at line 1) near <eof>"
        );
        assert_eq!(err("syntax error"), "c:1: syntax error near 'error'");
        assert_eq!(err("1.000"), "c:1: unexpected symbol near '1.000'");
        assert_eq!(err("x = 'abc"), "c:1: unfinished string near <eof>");
        assert_eq!(err("x = 3x"), "c:1: malformed number near '3x'");
        assert_eq!(err("return;;"), "c:1: <eof> expected near ';'");
        assert_eq!(err("\u{1}a = 1"), "c:1: unexpected symbol near '<\\1>'");
        assert_eq!(
            err("goto f; local x; ::f:: print(x)"),
            "c:1: <goto f> at line 1 jumps into the scope of local 'x'"
        );
        assert_eq!(err("break"), "c:1: break outside loop at line 1");
        assert_eq!(
            err("local x <const> = 1; x = 2"),
            "c:1: attempt to assign to const variable 'x'"
        );
        assert_eq!(err("f() = 1"), "c:1: syntax error near '='");
    }

    #[test]
    fn break_may_be_followed() {
        let checked = check(b"while x do break; x = 1 end", 2).expect("parses");
        assert_eq!(checked.breaks, vec![11]);
        let checked = check(b"while x do break end", 2).expect("parses");
        assert!(checked.breaks.is_empty());
    }
}
