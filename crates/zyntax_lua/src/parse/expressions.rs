//! Expressions support.

use super::*;

pub(super) fn show(name: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(name)
}

impl<'a> Parser<'a> {
    pub(super) fn explist(&mut self) -> R<Exp<'a>> {
        let mut e = self.expr()?;
        while self.test_next(T::Char(b','))? {
            self.put_in_next_reg(&e)?;
            e = self.expr()?;
        }
        Ok(e)
    }

    pub(super) fn field_sel(&mut self) -> R<()> {
        self.next()?;
        self.check_name()?;
        Ok(())
    }

    pub(super) fn index(&mut self) -> R<()> {
        self.next()?;
        self.expr()?;
        self.check_next(T::Char(b']'))
    }

    pub(super) fn body(&mut self, method: bool, line: usize) -> R<()> {
        self.open_func(line);
        self.check_next(T::Char(b'('))?;
        if method {
            self.new_local(b"self")?;
            self.adjust_locals(1);
        }
        self.parlist()?;
        self.check_next(T::Char(b')'))?;
        self.statlist()?;
        self.check_match(T::End, T::Function, line)?;
        self.close_func()
    }

    pub(super) fn parlist(&mut self) -> R<()> {
        let mut nparams = 0;
        let mut vararg = false;
        if self.t() != T::Char(b')') {
            loop {
                match self.t() {
                    T::Name => {
                        let name = self.check_name()?;
                        self.new_local(name)?;
                        nparams += 1;
                    }
                    T::Dots => {
                        self.next()?;
                        vararg = true;
                    }
                    _ => return Err(self.syntax_error("<name> or '...' expected")),
                }
                if vararg || !self.test_next(T::Char(b','))? {
                    break;
                }
            }
        }
        self.adjust_locals(nparams);
        let fs = self.fs();
        fs.vararg = vararg;
        let n = fs.nactvar;
        self.reserve(n)
    }

    pub(super) fn constructor(&mut self) -> R<()> {
        let line = self.lx.line;
        self.reserve(1)?;
        let table_reg = self.fs().free_reg;
        self.check_next(T::Char(b'{'))?;
        let mut pending = Exp::of(Ek::Void);
        let mut to_store = 0;
        loop {
            if self.t() == T::Char(b'}') {
                break;
            }
            if pending.k != Ek::Void {
                self.put_in_next_reg(&pending)?;
                pending = Exp::of(Ek::Void);
                if to_store == FIELDS_PER_FLUSH {
                    self.fs().free_reg = table_reg;
                    to_store = 0;
                }
            }
            let list = match self.t() {
                T::Name => self.lx.lookahead()? != T::Char(b'='),
                T::Char(b'[') => false,
                _ => true,
            };
            if list {
                pending = self.expr()?;
                to_store += 1;
            } else {
                let reg = self.fs().free_reg;
                if self.t() == T::Name {
                    self.check_name()?;
                } else {
                    self.index()?;
                }
                self.check_next(T::Char(b'='))?;
                self.expr()?;
                self.fs().free_reg = reg;
            }
            if !(self.test_next(T::Char(b','))? || self.test_next(T::Char(b';'))?) {
                break;
            }
        }
        self.check_match(T::Char(b'}'), T::Char(b'{'), line)
    }

    /// The arguments of a call whose function is in register `base`.
    pub(super) fn func_args(&mut self, line: usize, base: usize) -> R<()> {
        match self.t() {
            T::Char(b'(') => {
                self.next()?;
                if self.t() != T::Char(b')') {
                    let e = self.explist()?;
                    if !e.multiple() {
                        self.put_in_next_reg(&e)?;
                    }
                }
                self.check_match(T::Char(b')'), T::Char(b'('), line)?;
            }
            T::Char(b'{') => self.constructor()?,
            T::Str => self.next()?,
            _ => return Err(self.syntax_error("function arguments expected")),
        }
        self.fs().free_reg = base + 1;
        Ok(())
    }

    pub(super) fn primary_exp(&mut self) -> R<Exp<'a>> {
        match self.t() {
            T::Char(b'(') => {
                let line = self.lx.line;
                self.next()?;
                self.expr()?;
                self.check_match(T::Char(b')'), T::Char(b'('), line)?;
                Ok(Exp::of(Ek::Value))
            }
            T::Name => self.single_var(),
            _ => Err(self.syntax_error("unexpected symbol")),
        }
    }

    pub(super) fn suffixed_exp(&mut self) -> R<Exp<'a>> {
        let mut v = self.primary_exp()?;
        loop {
            match self.t() {
                T::Char(b'.') => {
                    self.field_sel()?;
                    v = Exp::of(Ek::Indexed);
                }
                T::Char(b'[') => {
                    self.index()?;
                    v = Exp::of(Ek::Indexed);
                }
                T::Char(b':') => {
                    let line = self.lx.line;
                    self.next()?;
                    self.check_name()?;
                    let base = self.fs().free_reg;
                    self.reserve(2)?;
                    self.func_args(line, base)?;
                    v = Exp::of(Ek::Call);
                }
                T::Char(b'(' | b'{') | T::Str => {
                    let line = self.lx.line;
                    let base = self.fs().free_reg;
                    self.put_in_next_reg(&v)?;
                    self.func_args(line, base)?;
                    v = Exp::of(Ek::Call);
                }
                _ => return Ok(v),
            }
        }
    }

    pub(super) fn simple_exp(&mut self) -> R<Exp<'a>> {
        let e = match self.t() {
            T::Flt | T::Int | T::Str | T::Nil | T::True | T::False => Exp::of(Ek::Value),
            T::Dots => {
                if !self.funcs.last().expect("a function").vararg {
                    return Err(self.syntax_error("cannot use '...' outside a vararg function"));
                }
                Exp::of(Ek::Vararg)
            }
            T::Char(b'{') => {
                self.constructor()?;
                return Ok(Exp::of(Ek::Value));
            }
            T::Function => {
                self.next()?;
                let line = self.lx.line;
                self.body(false, line)?;
                return Ok(Exp::of(Ek::Value));
            }
            _ => return self.suffixed_exp(),
        };
        self.next()?;
        Ok(e)
    }

    pub(super) fn expr(&mut self) -> R<Exp<'a>> {
        self.subexpr(0).map(|(e, _)| e)
    }

    /// `subexpr`: operators binding tighter than `limit`, and the first
    /// operator it left.
    pub(super) fn subexpr(&mut self, limit: u8) -> R<(Exp<'a>, Option<(u8, u8)>)> {
        self.enter_level()?;
        let mut e = if matches!(self.t(), T::Not | T::Char(b'-' | b'~' | b'#')) {
            self.next()?;
            self.subexpr(UNARY_PRIORITY)?;
            Exp::of(Ek::Value)
        } else {
            self.simple_exp()?
        };
        let mut op = binary_priority(self.t());
        while let Some((left, right)) = op {
            if left <= limit {
                break;
            }
            self.next()?;
            let (_, next) = self.subexpr(right)?;
            e = Exp::of(Ek::Value);
            op = next;
        }
        self.leave_level();
        Ok((e, op))
    }
}

/// A binary operator's left and right priority, from `lparser.c`.
pub(super) fn binary_priority(t: T) -> Option<(u8, u8)> {
    Some(match t {
        T::Char(b'+' | b'-') => (10, 10),
        T::Char(b'*' | b'%' | b'/') | T::IDiv => (11, 11),
        T::Char(b'^') => (14, 13),
        T::Char(b'&') => (6, 6),
        T::Char(b'|') => (4, 4),
        T::Char(b'~') => (5, 5),
        T::Shl | T::Shr => (7, 7),
        T::Concat => (9, 8),
        T::Eq | T::Char(b'<' | b'>') | T::Le | T::Ne | T::Ge => (3, 3),
        T::And => (2, 2),
        T::Or => (1, 1),
        _ => return None,
    })
}
