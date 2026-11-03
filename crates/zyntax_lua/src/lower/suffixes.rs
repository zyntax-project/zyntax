//! Suffixes for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A prefix followed by suffixes: indexing and calls, left to
    /// right. The last suffix may produce several values.
    pub(super) fn suffixed(
        &mut self,
        prefix: &Prefix,
        suffixes: &[&Suffix],
        span: Span,
    ) -> Result<Multi> {
        // `_G.name` is the global.
        if let Some(name) = self.typer().global_member(prefix, suffixes) {
            let v = self.read_global(&name, span)?;
            return Ok(Multi::Fixed(vec![v]));
        }
        if let Some(name) = self
            .typer()
            .global_member(prefix, &suffixes[..suffixes.len().min(1)])
        {
            let mut multi = None;
            let head = self.read_global(&name, span)?;
            let desc = self.describe_chain(prefix, &suffixes[..1]);
            return self.suffixes_from(head, desc, suffixes, 1, &mut multi, None, span);
        }
        // A direct call to a known function or a builtin, possibly
        // followed by more suffixes on its result: only the last call
        // of a `return` is a tail call.
        let (mut multi, first) = self.suffixed_head(prefix, suffixes, span)?;
        if first == suffixes.len()
            && let Some(m) = multi
        {
            return Ok(m);
        }
        let (current, desc) = match multi.take() {
            Some(m) => (self.first_of(m, span), None),
            None => match prefix {
                Prefix::Name(token) => (self.read_name(token)?, self.describe_name(token)),
                Prefix::Expression(e) => (self.expr(e)?, self.describe(e)),
                _ => return unsupported("this prefix", span),
            },
        };
        let exact = (first == 0).then(|| self.exact_prefix(prefix)).flatten();
        self.suffixes_from(current, desc, suffixes, first, &mut multi, exact, span)
    }

    /// The first suffix of a chain applied as a direct call to a known
    /// function or a builtin, when it is one: the call and how many
    /// suffixes it took.
    pub(super) fn suffixed_head(
        &mut self,
        prefix: &Prefix,
        suffixes: &[&Suffix],
        span: Span,
    ) -> Result<(Option<Multi>, usize)> {
        let mut multi: Option<Multi> = None;
        let mut first = 0;
        let tail = self.tail_span;
        if let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes.first()
            && let Some(f) = self.typer().known_callee(prefix)
        {
            if suffixes.len() > 1 {
                self.tail_span = None;
            }
            // A global declared once as this function is the function,
            // and so is a local that holds a function from its
            // declaration on; any other name holds its value, which may
            // be nil by now.
            let declared = match prefix {
                Prefix::Name(token) => match self.scopes().binding(token) {
                    Some(Binding::Global(name)) => {
                        self.scopes().known_global_function(name) == Some(f)
                    }
                    Some(Binding::Local(v) | Binding::Upvalue(v)) => {
                        self.scopes().always_function(*v)
                    }
                    _ => false,
                },
                _ => false,
            };
            if declared {
                // The record holds the captures, when there are any.
                let record = match prefix {
                    Prefix::Name(token) if self.m.takes_env(f) => Some(self.read_name(token)?),
                    _ => None,
                };
                let desc = match prefix {
                    Prefix::Name(token) => self.describe_name(token),
                    _ => None,
                };
                multi = Some(self.direct_call(f, record, None, args, desc, span)?);
            } else {
                let callee = match prefix {
                    Prefix::Name(token) => self.read_name(token)?,
                    Prefix::Expression(e) => self.expr(e)?,
                    _ => return unsupported("this prefix", span),
                };
                let desc = match prefix {
                    Prefix::Name(token) => self.describe_name(token),
                    Prefix::Expression(e) => self.describe(e),
                    _ => None,
                };
                let returns = self.m.sig(f).returns.clone();
                let (pre, vals, tail) = self.call_values(None, args, span)?;
                multi = Some(self.dispatch(
                    callee,
                    Guard::NotNil,
                    &[f],
                    returns,
                    pre,
                    vals,
                    tail,
                    desc,
                    span,
                ));
            }
            self.tail_span = tail;
            first = 1;
        }
        if multi.is_none() {
            let head = suffixes.len().min(2);
            for n in (1..=head).rev() {
                if let Some(b) = self.typer().builtin_callee(prefix, &suffixes[..n]) {
                    let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes[..n].last()
                    else {
                        unreachable!("a builtin callee ends in a call");
                    };
                    if suffixes.len() > n {
                        self.tail_span = None;
                    }
                    let called = self.builtin_call(b, None, args, span);
                    self.tail_span = tail;
                    multi = Some(called?);
                    first = n;
                    break;
                }
            }
        }
        Ok((multi, first))
    }

    /// The suffixes from `first` on, applied to `current`, which `desc`
    /// names.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn suffixes_from(
        &mut self,
        mut current: Val,
        mut desc: Desc,
        suffixes: &[&Suffix],
        first: usize,
        multi: &mut Option<Multi>,
        mut exact: Option<ShapeId>,
        span: Span,
    ) -> Result<Multi> {
        let mut called = false;
        for (i, s) in suffixes.iter().enumerate().skip(first) {
            let receiver_class = exact.take();
            // The call a field call's lookup made already.
            if std::mem::take(&mut called) {
                if i + 1 == suffixes.len() {
                    return Ok(multi.take().expect("a field call's results"));
                }
                continue;
            }
            if let Some(m) = multi.take() {
                current = self.first_of(m, span);
                desc = None;
            }
            let last = i + 1 == suffixes.len();
            if let Suffix::Index(ast::Index::Dot { name, .. }) = s
                && let Some(next) = suffixes.get(i + 1)
                && let Some(m) = self.field_call(
                    &current,
                    &ident(name),
                    next,
                    desc.clone(),
                    receiver_class,
                    span,
                )?
            {
                *multi = Some(m);
                called = true;
                desc = None;
                continue;
            }
            match s {
                Suffix::Index(ast::Index::Dot { name, .. }) => {
                    let key = Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    };
                    current = self.field_read(current, key, &ident(name), desc, span);
                    desc = Self::describe_suffix(s);
                }
                Suffix::Index(ast::Index::Brackets { expression, .. }) => {
                    let key = self.expr(expression)?;
                    current = match crate::scope::literal_string(expression) {
                        Some(name) => self.field_read(current, key, &name, desc, span),
                        None => self.index_read(current, key, desc, span),
                    };
                    desc = Self::describe_suffix(s);
                }
                Suffix::Call(ast::Call::AnonymousCall(args)) => {
                    let tail = self.tail_span.take_if(|_| !last);
                    let m = self.value_call(current.clone(), None, args, desc, span);
                    self.tail_span = self.tail_span.or(tail);
                    let m = m?;
                    if last {
                        return Ok(m);
                    }
                    *multi = Some(m);
                    desc = None;
                }
                Suffix::Call(ast::Call::MethodCall(mc)) => {
                    let returned = (!last)
                        .then(|| {
                            self.returned_receiver(current.ty, &ident(mc.name()), receiver_class)
                        })
                        .flatten();
                    let tail = self.tail_span.take_if(|_| !last);
                    let m = self.method_call(current.clone(), mc, desc, receiver_class, span);
                    self.tail_span = self.tail_span.or(tail);
                    let m = m?;
                    if last {
                        return Ok(m);
                    }
                    *multi = Some(m);
                    exact = returned;
                    desc = None;
                }
                _ => return unsupported("this suffix", span),
            }
        }
        Ok(Multi::Fixed(vec![current]))
    }

    /// `obj.name(args)` for a receiver of a shape whose lookups of
    /// `name` all end in slots holding functions the types resolve the
    /// call to: called directly, by the end the lookup takes. None for
    /// any other call; `desc` names `obj`.
    pub(super) fn field_call(
        &mut self,
        obj: &Val,
        name: &str,
        call_suffix: &Suffix,
        desc: Desc,
        exact: Option<ShapeId>,
        span: Span,
    ) -> Result<Option<Multi>> {
        let (Ty::Shape(k), Suffix::Call(ast::Call::AnonymousCall(args))) = (obj.ty, call_suffix)
        else {
            return Ok(None);
        };
        let (returns, funcs) = match self.typer().field_ty(Ty::Shape(k), name).settled() {
            Ty::Func(f) => (self.m.sig(f).returns.clone(), vec![f]),
            _ => {
                let Some(targets) = self.typer().method_targets(k, name) else {
                    return Ok(None);
                };
                let returns = self.typer().method_returns(Ty::Shape(k), name).settled();
                let funcs = targets
                    .iter()
                    .filter_map(|t| match t {
                        Ty::Func(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                (returns, funcs)
            }
        };
        let Some(which) = self.finder(k, name, FinderMode::Which, span) else {
            return Ok(None);
        };
        let known = which.ends.iter().all(|(_, end)| match end {
            End::Slot(Ty::Func(f)) => funcs.contains(f),
            _ => false,
        });
        if !known {
            return Ok(None);
        }
        self.which_call(
            obj.clone(),
            k,
            name,
            CallKind::Field,
            exact,
            returns,
            args,
            desc,
            span,
        )
    }
}
