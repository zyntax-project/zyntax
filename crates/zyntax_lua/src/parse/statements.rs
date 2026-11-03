//! Statements support.

use super::*;

impl<'a> Parser<'a> {
    pub(super) fn main(&mut self) -> R<()> {
        self.open_func(0);
        let fs = self.fs();
        fs.vararg = true;
        fs.upvalues.push((ENV, Kind::Regular));
        self.next()?;
        self.statlist()?;
        self.check(T::Eos)?;
        self.close_func()
    }

    pub(super) fn statlist(&mut self) -> R<()> {
        while !self.block_follow(true) {
            if self.t() == T::Return {
                return self.statement();
            }
            self.statement()?;
        }
        Ok(())
    }

    pub(super) fn block(&mut self) -> R<()> {
        self.enter_block(false);
        self.statlist()?;
        self.leave_block()
    }

    pub(super) fn statement(&mut self) -> R<()> {
        let line = self.lx.line;
        self.enter_level()?;
        match self.t() {
            T::Char(b';') => self.next()?,
            T::If => self.if_stat(line)?,
            T::While => self.while_stat(line)?,
            T::Do => {
                self.next()?;
                self.block()?;
                self.check_match(T::End, T::Do, line)?;
            }
            T::For => self.for_stat(line)?,
            T::Repeat => self.repeat_stat(line)?,
            T::Function => self.func_stat(line)?,
            T::Local => {
                self.next()?;
                if self.test_next(T::Function)? {
                    self.local_func()?;
                } else {
                    self.local_stat()?;
                }
            }
            T::DbColon => {
                self.next()?;
                let name = self.check_name()?;
                self.label_stat(name, line)?;
            }
            T::Return => {
                self.next()?;
                self.ret_stat()?;
            }
            T::Break => {
                let at = self.lx.tok.at;
                let line = self.lx.line;
                self.next()?;
                let nactvar = self.fs().nactvar;
                self.gotos.push(Label {
                    name: BREAK,
                    line,
                    nactvar,
                });
                if !self.block_follow(true) {
                    self.checked.breaks.push(at);
                }
            }
            T::Goto => {
                self.next()?;
                self.goto_stat()?;
            }
            _ => self.expr_stat()?,
        }
        let fs = self.fs();
        fs.free_reg = fs.nactvar;
        self.leave_level();
        Ok(())
    }

    pub(super) fn if_stat(&mut self, line: usize) -> R<()> {
        self.test_then_block()?;
        while self.t() == T::Elseif {
            self.test_then_block()?;
        }
        if self.test_next(T::Else)? {
            self.block()?;
        }
        self.check_match(T::End, T::If, line)
    }

    pub(super) fn test_then_block(&mut self) -> R<()> {
        self.next()?;
        self.expr()?;
        self.check_next(T::Then)?;
        self.enter_block(false);
        self.statlist()?;
        self.leave_block()
    }

    pub(super) fn while_stat(&mut self, line: usize) -> R<()> {
        self.next()?;
        self.expr()?;
        self.enter_block(true);
        self.check_next(T::Do)?;
        self.block()?;
        self.check_match(T::End, T::While, line)?;
        self.leave_block()
    }

    pub(super) fn repeat_stat(&mut self, line: usize) -> R<()> {
        self.enter_block(true);
        self.enter_block(false);
        self.next()?;
        self.statlist()?;
        self.check_match(T::Until, T::Repeat, line)?;
        self.expr()?;
        self.leave_block()?;
        self.leave_block()
    }

    pub(super) fn exp1(&mut self) -> R<()> {
        let e = self.expr()?;
        self.put_in_next_reg(&e)
    }

    pub(super) fn for_stat(&mut self, line: usize) -> R<()> {
        self.enter_block(true);
        self.next()?;
        let name = self.check_name()?;
        match self.t() {
            T::Char(b'=') => {
                for _ in 0..3 {
                    self.new_local(b"(for state)")?;
                }
                self.new_local(name)?;
                self.check_next(T::Char(b'='))?;
                self.exp1()?;
                self.check_next(T::Char(b','))?;
                self.exp1()?;
                if self.test_next(T::Char(b','))? {
                    self.exp1()?;
                } else {
                    self.reserve(1)?;
                }
                self.adjust_locals(3);
                self.for_body(1)?;
            }
            T::Char(b',') | T::In => {
                for _ in 0..4 {
                    self.new_local(b"(for state)")?;
                }
                self.new_local(name)?;
                let mut nvars = 5;
                while self.test_next(T::Char(b','))? {
                    let name = self.check_name()?;
                    self.new_local(name)?;
                    nvars += 1;
                }
                self.check_next(T::In)?;
                self.explist()?;
                self.adjust_locals(4);
                self.for_body(nvars - 4)?;
            }
            _ => return Err(self.syntax_error("'=' or 'in' expected")),
        }
        self.check_match(T::End, T::For, line)?;
        self.leave_block()
    }

    pub(super) fn for_body(&mut self, nvars: usize) -> R<()> {
        self.check_next(T::Do)?;
        self.enter_block(false);
        self.adjust_locals(nvars);
        self.block()?;
        self.leave_block()
    }

    pub(super) fn func_stat(&mut self, line: usize) -> R<()> {
        self.next()?;
        let mut v = self.single_var()?;
        let mut method = false;
        while self.t() == T::Char(b'.') {
            self.field_sel()?;
            v = Exp::of(Ek::Indexed);
        }
        if self.t() == T::Char(b':') {
            method = true;
            self.field_sel()?;
            v = Exp::of(Ek::Indexed);
        }
        self.body(method, line)?;
        self.check_readonly(&v)
    }

    pub(super) fn local_func(&mut self) -> R<()> {
        let name = self.check_name()?;
        self.new_local(name)?;
        self.adjust_locals(1);
        let line = self.lx.line;
        self.body(false, line)
    }

    pub(super) fn local_stat(&mut self) -> R<()> {
        let mut to_close = false;
        let mut nvars = 0;
        loop {
            let name = self.check_name()?;
            self.new_local(name)?;
            let kind = self.local_attribute()?;
            self.actvar.last_mut().expect("the local").kind = kind;
            if kind == Kind::Close {
                if to_close {
                    return Err(self.sem_error("multiple to-be-closed variables in local list"));
                }
                to_close = true;
            }
            nvars += 1;
            if !self.test_next(T::Char(b','))? {
                break;
            }
        }
        if self.test_next(T::Char(b'='))? {
            self.explist()?;
        }
        self.adjust_locals(nvars);
        Ok(())
    }

    pub(super) fn local_attribute(&mut self) -> R<Kind> {
        if !self.test_next(T::Char(b'<'))? {
            return Ok(Kind::Regular);
        }
        let attr = self.check_name()?;
        self.check_next(T::Char(b'>'))?;
        match attr {
            b"const" => Ok(Kind::Const),
            b"close" => Ok(Kind::Close),
            other => Err(self.sem_error(&format!("unknown attribute '{}'", show(other)))),
        }
    }

    pub(super) fn label_stat(&mut self, name: &'a [u8], line: usize) -> R<()> {
        self.check_next(T::DbColon)?;
        while matches!(self.t(), T::Char(b';') | T::DbColon) {
            self.statement()?;
        }
        if let Some(lb) = self.find_label(name) {
            return Err(self.sem_error(&format!(
                "label '{}' already defined on line {}",
                show(name),
                lb.line
            )));
        }
        let last = self.block_follow(false);
        self.create_label(name, line, last)
    }

    pub(super) fn goto_stat(&mut self) -> R<()> {
        let line = self.lx.line;
        let name = self.check_name()?;
        if self.find_label(name).is_none() {
            let nactvar = self.fs().nactvar;
            self.gotos.push(Label {
                name,
                line,
                nactvar,
            });
        }
        Ok(())
    }

    pub(super) fn ret_stat(&mut self) -> R<()> {
        if !self.block_follow(true) && self.t() != T::Char(b';') {
            self.explist()?;
        }
        self.test_next(T::Char(b';'))?;
        Ok(())
    }

    pub(super) fn expr_stat(&mut self) -> R<()> {
        let v = self.suffixed_exp()?;
        if matches!(self.t(), T::Char(b'=' | b',')) {
            self.rest_assign(v)
        } else if v.k != Ek::Call {
            Err(self.syntax_error("syntax error"))
        } else {
            Ok(())
        }
    }

    pub(super) fn rest_assign(&mut self, lh: Exp<'a>) -> R<()> {
        if !lh.is_var() {
            return Err(self.syntax_error("syntax error"));
        }
        self.check_readonly(&lh)?;
        if self.test_next(T::Char(b','))? {
            let nv = self.suffixed_exp()?;
            self.enter_level()?;
            self.rest_assign(nv)?;
            self.leave_level();
        } else {
            self.check_next(T::Char(b'='))?;
            self.explist()?;
        }
        Ok(())
    }
}
