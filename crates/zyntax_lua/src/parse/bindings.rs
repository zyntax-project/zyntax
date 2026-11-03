//! Bindings support.

use super::*;

impl<'a> Parser<'a> {
    pub(super) fn t(&self) -> T {
        self.lx.tok.t
    }

    pub(super) fn fs(&mut self) -> &mut Func<'a> {
        self.funcs.last_mut().expect("a function")
    }

    pub(super) fn syntax_error(&self, msg: &str) -> SyntaxError {
        self.lx.error(msg, Some(self.t()))
    }

    /// `luaK_semerror`: a message without the token.
    pub(super) fn sem_error(&self, msg: &str) -> SyntaxError {
        self.lx.error(msg, None)
    }

    pub(super) fn error_expected(&self, t: T) -> SyntaxError {
        self.syntax_error(&format!("{} expected", token_str(t)))
    }

    pub(super) fn error_limit(&self, limit: usize, what: &str) -> SyntaxError {
        let line = self.funcs.last().map_or(0, |f| f.line_defined);
        let place = if line == 0 {
            "main function".to_string()
        } else {
            format!("function at line {line}")
        };
        self.syntax_error(&format!("too many {what} (limit is {limit}) in {place}"))
    }

    pub(super) fn enter_level(&mut self) -> R<()> {
        self.level += 1;
        if self.level >= MAX_C_CALLS {
            return Err(SyntaxError {
                line: None,
                message: "C stack overflow".to_string(),
                offset: self.lx.pos,
            });
        }
        Ok(())
    }

    pub(super) fn leave_level(&mut self) {
        self.level -= 1;
    }

    pub(super) fn next(&mut self) -> R<()> {
        self.lx.next()
    }

    pub(super) fn test_next(&mut self, t: T) -> R<bool> {
        if self.t() == t {
            self.next()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(super) fn check(&self, t: T) -> R<()> {
        if self.t() != t {
            return Err(self.error_expected(t));
        }
        Ok(())
    }

    pub(super) fn check_next(&mut self, t: T) -> R<()> {
        self.check(t)?;
        self.next()
    }

    pub(super) fn check_match(&mut self, what: T, who: T, line: usize) -> R<()> {
        if self.test_next(what)? {
            return Ok(());
        }
        if line == self.lx.line {
            return Err(self.error_expected(what));
        }
        Err(self.syntax_error(&format!(
            "{} expected (to close {} at line {line})",
            token_str(what),
            token_str(who)
        )))
    }

    pub(super) fn check_name(&mut self) -> R<&'a [u8]> {
        self.check(T::Name)?;
        let name = self.lx.tok.name;
        self.next()?;
        Ok(name)
    }

    pub(super) fn reserve(&mut self, n: usize) -> R<()> {
        let fs = self.fs();
        fs.free_reg += n;
        if fs.free_reg >= MAX_REGS {
            return Err(self.syntax_error("function or expression needs too many registers"));
        }
        Ok(())
    }

    /// `luaK_exp2nextreg`: the expression takes the next register.
    pub(super) fn put_in_next_reg(&mut self, e: &Exp) -> R<()> {
        if e.k == Ek::Void {
            return Ok(());
        }
        self.reserve(1)
    }

    pub(super) fn block_follow(&self, with_until: bool) -> bool {
        match self.t() {
            T::Else | T::Elseif | T::End | T::Eos => true,
            T::Until => with_until,
            _ => false,
        }
    }

    pub(super) fn new_local(&mut self, name: &'a [u8]) -> R<()> {
        let first = self.funcs.last().expect("a function").first_local;
        if self.actvar.len() + 1 - first > MAX_VARS {
            return Err(self.error_limit(MAX_VARS, "local variables"));
        }
        self.actvar.push(Var {
            name,
            kind: Kind::Regular,
        });
        Ok(())
    }

    pub(super) fn adjust_locals(&mut self, n: usize) {
        self.fs().nactvar += n;
    }

    pub(super) fn enter_block(&mut self, is_loop: bool) {
        let (labels, gotos) = (self.labels.len(), self.gotos.len());
        let fs = self.fs();
        let nactvar = fs.nactvar;
        fs.blocks.push(Block {
            first_label: labels,
            first_goto: gotos,
            nactvar,
            is_loop,
        });
    }

    pub(super) fn leave_block(&mut self) -> R<()> {
        let fs = self.funcs.last().expect("a function");
        let bl = fs.blocks.last().expect("a block");
        let (bl_nactvar, first_label, first_goto, is_loop) =
            (bl.nactvar, bl.first_label, bl.first_goto, bl.is_loop);
        let removed = fs.nactvar - bl_nactvar;
        self.actvar.truncate(self.actvar.len() - removed);
        self.fs().nactvar = bl_nactvar;
        if is_loop {
            self.create_label(BREAK, 0, false)?;
        }
        self.labels.truncate(first_label);
        let fs = self.fs();
        fs.blocks.pop();
        if !fs.blocks.is_empty() {
            for gt in &mut self.gotos[first_goto..] {
                gt.nactvar = bl_nactvar;
            }
        } else if let Some(gt) = self.gotos.get(first_goto).copied() {
            let msg = if gt.name == BREAK {
                format!("break outside loop at line {}", gt.line)
            } else {
                format!(
                    "no visible label '{}' for <goto> at line {}",
                    show(gt.name),
                    gt.line
                )
            };
            return Err(self.sem_error(&msg));
        }
        Ok(())
    }

    pub(super) fn find_label(&self, name: &[u8]) -> Option<Label<'a>> {
        let first = self.funcs.last().expect("a function").first_label;
        self.labels[first..]
            .iter()
            .find(|l| l.name == name)
            .copied()
    }

    pub(super) fn create_label(&mut self, name: &'a [u8], line: usize, last: bool) -> R<()> {
        let fs = self.funcs.last().expect("a function");
        let bl = fs.blocks.last().expect("a block");
        let nactvar = if last { bl.nactvar } else { fs.nactvar };
        let (first_local, first_goto) = (fs.first_local, bl.first_goto);
        self.labels.push(Label {
            name,
            line,
            nactvar,
        });
        let mut i = first_goto;
        while i < self.gotos.len() {
            let gt = self.gotos[i];
            if gt.name != name {
                i += 1;
                continue;
            }
            if gt.nactvar < nactvar {
                let var = show(self.actvar[first_local + gt.nactvar].name);
                let msg = format!(
                    "<goto {}> at line {} jumps into the scope of local '{var}'",
                    show(gt.name),
                    gt.line
                );
                return Err(self.sem_error(&msg));
            }
            self.gotos.remove(i);
        }
        Ok(())
    }

    pub(super) fn open_func(&mut self, line_defined: usize) {
        self.funcs.push(Func {
            first_local: self.actvar.len(),
            first_label: self.labels.len(),
            nactvar: 0,
            upvalues: Vec::new(),
            blocks: Vec::new(),
            vararg: false,
            line_defined,
            free_reg: 0,
        });
        self.enter_block(false);
    }

    pub(super) fn close_func(&mut self) -> R<()> {
        self.leave_block()?;
        self.funcs.pop();
        Ok(())
    }

    /// The variable `name` is at level `f`: a local, an upvalue (made in
    /// every function between), or `None` for a global.
    pub(super) fn resolve(&mut self, f: usize, name: &'a [u8]) -> R<Option<Ek>> {
        let fs = &self.funcs[f];
        for i in (0..fs.nactvar).rev() {
            let var = &self.actvar[fs.first_local + i];
            if var.name == name {
                return Ok(Some(Ek::Local(var.kind)));
            }
        }
        if let Some((_, kind)) = fs.upvalues.iter().find(|(n, _)| *n == name) {
            return Ok(Some(Ek::Upvalue(*kind)));
        }
        if f == 0 {
            return Ok(None);
        }
        let kind = match self.resolve(f - 1, name)? {
            Some(Ek::Local(kind) | Ek::Upvalue(kind)) => kind,
            _ => return Ok(None),
        };
        if self.funcs[f].upvalues.len() + 1 > MAX_UPVALUES {
            let line = self.funcs[f].line_defined;
            let place = if line == 0 {
                "main function".to_string()
            } else {
                format!("function at line {line}")
            };
            return Err(self.syntax_error(&format!(
                "too many upvalues (limit is {MAX_UPVALUES}) in {place}"
            )));
        }
        self.funcs[f].upvalues.push((name, kind));
        Ok(Some(Ek::Upvalue(kind)))
    }

    pub(super) fn single_var(&mut self) -> R<Exp<'a>> {
        let name = self.check_name()?;
        let f = self.funcs.len() - 1;
        match self.resolve(f, name)? {
            Some(k) => Ok(Exp { k, name }),
            None => {
                self.resolve(f, ENV)?;
                Ok(Exp {
                    k: Ek::Indexed,
                    name,
                })
            }
        }
    }

    pub(super) fn check_readonly(&self, e: &Exp) -> R<()> {
        match e.k {
            Ek::Local(k) | Ek::Upvalue(k) if k != Kind::Regular => Err(self.sem_error(&format!(
                "attempt to assign to const variable '{}'",
                show(e.name)
            ))),
            _ => Ok(()),
        }
    }
}
