//! A parsing machine for `GrammarIR`.
//!
//! The interpreter walks `PatternIR` trees on every parse, so each node
//! costs an enum dispatch and a recursive call, and backtracking rides
//! on the Rust call stack. This module compiles the same patterns once,
//! when a grammar is loaded, into a flat instruction sequence executed
//! by a loop with explicit backtrack and call stacks.
//!
//! Compilation happens in memory at load time. Nothing is generated as
//! source, so a grammar stays something a program reads at runtime.
//!
//! The instruction set is the one a PEG needs and no more: match a
//! terminal, call a rule, manage the choice points that make ordered
//! choice and repetition work, and shape the value a rule returns.
//! `Choice` pushes a backtrack entry, `Commit` drops it, and a failure
//! unwinds to the most recent one.
//!
//! Values are built in a single register: a terminal writes what it
//! matched, a sequence leaves the last element's value behind, and a
//! repetition accumulates into a list stack. Rules whose form the
//! compiler does not cover run on the interpreter, so a grammar always
//! parses whether or not the machine can express all of it.

use super::interpreter::GrammarInterpreter;
use super::memo::MemoEntry;
use super::state::{ParseResult, ParsedValue, ParserState};
use crate::grammar::{CharClass, GrammarIR, PatternIR, RuleIR, RuleModifier};
use std::collections::HashMap;
use zyntax_typed_ast::Span;

/// One instruction. Positions are indices into [`Program::code`].
#[derive(Debug, Clone)]
pub enum Instr {
    /// Match a literal, or fail. `desc` is what a failure reports,
    /// built once here rather than formatted on every mismatch.
    Literal { text: String, desc: String },
    /// Match one character of a class, or fail.
    Class(CharClass),
    /// Match any one character, or fail at end of input.
    AnyChar,
    /// Match the start of input.
    Soi,
    /// Match the end of input.
    Eoi,
    /// Skip whitespace and comments.
    SkipWs,
    /// Set the value register to nothing.
    SetNone,
    /// Enter a rule.
    Call {
        rule: usize,
        /// Bind the rule's value under this name on success.
        binding: Option<String>,
    },
    /// Leave the current rule with the value register as its result.
    Ret,
    /// Push a backtrack entry that resumes at `alt` on failure.
    Choice { alt: usize },
    /// Drop the most recent backtrack entry and continue at `next`.
    Commit { next: usize },
    /// Fail, unwinding to the most recent backtrack entry.
    Fail,
    /// Bind the value register under a name.
    Bind(String),
    /// Wrap the value register as a present optional.
    OptSome,
    /// Set the value register to an absent optional, and bind it under
    /// `binding` so an action can still read the name.
    OptNone { binding: Option<String> },
    /// Open a list for a repetition.
    ListBegin,
    /// Jump to `body` when the open list is empty. Lets a repetition
    /// run its separator on every pass but the first.
    IfEmpty { body: usize },
    /// End one pass of a repetition: append the value register to the
    /// open list and go round again, or leave for `exit` when the pass
    /// consumed nothing or the list is full.
    RepeatEnd {
        body: usize,
        exit: usize,
        max: Option<usize>,
    },
    /// Close a repetition's list into the value register, failing if it
    /// is shorter than `min`.
    ListEnd { min: usize, binding: Option<String> },
    /// Begin a lookahead. `negative` inverts the sense; either way the
    /// position is restored when it ends. `end` is where a negative
    /// lookahead resumes once its inner pattern has failed.
    BeginLook { negative: bool, end: usize },
    /// End a lookahead begun by [`Instr::BeginLook`].
    EndLook { negative: bool },
}

/// One rule's place in a compiled program.
pub struct RuleSlot<'g> {
    /// Where the rule's code begins, or `None` for a rule the compiler
    /// could not express, which runs on the interpreter instead.
    pub entry: Option<usize>,
    /// The id this rule memoizes under, shared with the interpreter so
    /// both halves read and write the same packrat entries.
    pub memo_id: usize,
    pub atomic: bool,
    pub rule: &'g RuleIR,
}

/// A compiled grammar.
#[derive(Default)]
pub struct Program<'g> {
    /// Every rule's code, concatenated.
    pub code: Vec<Instr>,
    /// Every rule in the grammar, whether or not the machine runs it.
    pub rules: Vec<RuleSlot<'g>>,
    /// Where a rule name lands in [`Self::rules`].
    pub index_of: HashMap<&'g str, usize>,
    /// Rules the compiler could not express, by name. These keep
    /// running on the tree-walking interpreter, so a grammar using a
    /// form the machine does not cover still parses.
    pub unsupported: Vec<&'g str>,
}

impl<'g> Program<'g> {
    /// Instructions emitted, for reporting.
    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Rules the machine can run.
    pub fn supported(&self) -> usize {
        self.rules.len() - self.unsupported.len()
    }

    /// Fraction of rules the machine can run, as a percentage.
    pub fn coverage(&self) -> f64 {
        if self.rules.is_empty() {
            return 0.0;
        }
        100.0 * self.supported() as f64 / self.rules.len() as f64
    }

    /// Look up a rule by name.
    pub fn rule_index(&self, name: &str) -> Option<usize> {
        self.index_of.get(name).copied()
    }

    /// Whether the machine can run a rule by name.
    pub fn runs(&self, name: &str) -> bool {
        self.rule_index(name)
            .is_some_and(|i| self.rules[i].entry.is_some())
    }
}

/// Compile a grammar into a program.
///
/// `memo_ids` comes from the interpreter, so a rule the machine runs
/// and a rule the interpreter runs memoize under the same id and can
/// reuse each other's results.
///
/// A rule the compiler cannot express is recorded in
/// [`Program::unsupported`] rather than failing the whole grammar, so
/// coverage can grow one form at a time without the machine having to
/// be complete before it is useful.
pub fn compile<'g>(grammar: &'g GrammarIR, memo_ids: &HashMap<String, usize>) -> Program<'g> {
    let mut rule_index: Vec<(&'g String, &'g RuleIR)> = grammar.rules.iter().collect();
    // A stable order so entry indices are reproducible across runs.
    rule_index.sort_by(|a, b| a.0.cmp(b.0));

    // Rule indices have to exist before any body is compiled, because a
    // rule can call one that appears later.
    let mut index_of: HashMap<&'g str, usize> = HashMap::with_capacity(rule_index.len());
    for (i, (name, _)) in rule_index.iter().enumerate() {
        index_of.insert(name.as_str(), i);
    }

    let mut program = Program {
        code: Vec::new(),
        rules: Vec::with_capacity(rule_index.len()),
        index_of,
        unsupported: Vec::new(),
    };

    let mut bodies: Vec<Option<Vec<Instr>>> = Vec::with_capacity(rule_index.len());
    for (_, rule) in &rule_index {
        let atomic = rule.modifier == Some(RuleModifier::Atomic);
        let mut out = Vec::new();
        match emit(&rule.pattern, atomic, &program.index_of, &mut out) {
            Ok(()) => {
                out.push(Instr::Ret);
                bodies.push(Some(out));
            }
            Err(()) => bodies.push(None),
        }
    }

    for (i, (name, rule)) in rule_index.iter().enumerate() {
        let entry = match &bodies[i] {
            Some(body) => {
                let at = program.code.len();
                // Offsets inside a body are relative to its start.
                program.code.extend(body.iter().map(|i| rebase(i, at)));
                Some(at)
            }
            None => {
                program.unsupported.push(name.as_str());
                None
            }
        };
        program.rules.push(RuleSlot {
            entry,
            memo_id: memo_ids.get(*name).copied().unwrap_or(usize::MAX),
            atomic: rule.modifier == Some(RuleModifier::Atomic),
            rule,
        });
    }

    program
}

/// Shift a body-relative jump target to where the body was placed.
fn rebase(instr: &Instr, base: usize) -> Instr {
    let mut out = instr.clone();
    match &mut out {
        Instr::Choice { alt } => *alt += base,
        Instr::Commit { next } => *next += base,
        Instr::IfEmpty { body } => *body += base,
        Instr::RepeatEnd { body, exit, .. } => {
            *body += base;
            *exit += base;
        }
        Instr::BeginLook { end, .. } => *end += base,
        _ => {}
    }
    out
}

/// The binding a repetition or optional accumulates into.
///
/// Only a bound rule reference directly under the operator collects;
/// a binding nested deeper keeps whatever the last pass wrote, which
/// is what the interpreter does.
fn direct_binding(pattern: &PatternIR) -> Option<String> {
    match pattern {
        PatternIR::RuleRef {
            binding: Some(name),
            ..
        } => Some(name.clone()),
        _ => None,
    }
}

/// Emit code for one pattern, appending to `out`.
///
/// Returns `Err(())` for a form the machine does not cover yet, which
/// drops the whole rule to the interpreter rather than emitting
/// something that would parse differently.
fn emit(
    pattern: &PatternIR,
    atomic: bool,
    index_of: &HashMap<&str, usize>,
    out: &mut Vec<Instr>,
) -> Result<(), ()> {
    match pattern {
        PatternIR::Literal(s) => {
            out.push(Instr::Literal {
                text: s.clone(),
                desc: format!("'{}'", s),
            });
            Ok(())
        }
        PatternIR::CharClass(c) => {
            out.push(Instr::Class(c.clone()));
            Ok(())
        }
        PatternIR::Any => {
            out.push(Instr::AnyChar);
            Ok(())
        }
        PatternIR::StartOfInput => {
            out.push(Instr::Soi);
            Ok(())
        }
        PatternIR::EndOfInput => {
            out.push(Instr::Eoi);
            Ok(())
        }
        PatternIR::Whitespace => {
            out.push(Instr::SkipWs);
            out.push(Instr::SetNone);
            Ok(())
        }
        PatternIR::RuleRef { rule_name, binding } => {
            // The built-in names win over a grammar rule of the same
            // name, the order the interpreter resolves them in.
            match rule_name.as_str() {
                "SOI" => out.push(Instr::Soi),
                "EOI" => out.push(Instr::Eoi),
                "ANY" => out.push(Instr::AnyChar),
                "ASCII_DIGIT" | "ASCII_ALPHA" | "ASCII_ALPHANUMERIC" | "ASCII_HEX_DIGIT" => {
                    out.push(Instr::Class(CharClass::Builtin(rule_name.clone())))
                }
                name => {
                    let rule = *index_of.get(name).ok_or(())?;
                    out.push(Instr::Call {
                        rule,
                        binding: binding.clone(),
                    });
                    return Ok(());
                }
            }
            if let Some(name) = binding {
                out.push(Instr::Bind(name.clone()));
            }
            Ok(())
        }
        PatternIR::Sequence(items) => {
            if items.is_empty() {
                out.push(Instr::SetNone);
                return Ok(());
            }
            for (i, item) in items.iter().enumerate() {
                if !atomic && i > 0 {
                    out.push(Instr::SkipWs);
                }
                emit(item, atomic, index_of, out)?;
            }
            Ok(())
        }
        PatternIR::Choice(alts) => {
            if alts.is_empty() {
                out.push(Instr::Fail);
                return Ok(());
            }
            // Each alternative but the last is guarded by a choice
            // point: on failure the machine resumes at the next one.
            let mut commit_sites = Vec::new();
            for (i, alt) in alts.iter().enumerate() {
                let last = i + 1 == alts.len();
                let choice_site = if last {
                    None
                } else {
                    out.push(Instr::Choice { alt: usize::MAX });
                    Some(out.len() - 1)
                };
                emit(alt, atomic, index_of, out)?;
                if let Some(site) = choice_site {
                    out.push(Instr::Commit { next: usize::MAX });
                    commit_sites.push(out.len() - 1);
                    let here = out.len();
                    if let Instr::Choice { alt } = &mut out[site] {
                        *alt = here;
                    }
                }
            }
            let end = out.len();
            for site in commit_sites {
                if let Instr::Commit { next } = &mut out[site] {
                    *next = end;
                }
            }
            Ok(())
        }
        PatternIR::Optional(inner) => {
            out.push(Instr::Choice { alt: usize::MAX });
            let site = out.len() - 1;
            emit(inner, atomic, index_of, out)?;
            out.push(Instr::OptSome);
            out.push(Instr::Commit { next: usize::MAX });
            let commit_site = out.len() - 1;
            let absent = out.len();
            out.push(Instr::OptNone {
                binding: direct_binding(inner),
            });
            let end = out.len();
            if let Instr::Choice { alt } = &mut out[site] {
                *alt = absent;
            }
            if let Instr::Commit { next } = &mut out[commit_site] {
                *next = end;
            }
            Ok(())
        }
        PatternIR::Repeat {
            pattern,
            min,
            max,
            separator,
        } => {
            // Whitespace is skipped before every pass, including the
            // first, and the choice point is taken after it, so a pass
            // that fails leaves the position past that whitespace.
            out.push(Instr::ListBegin);
            let top = out.len();
            if !atomic {
                out.push(Instr::SkipWs);
            }
            out.push(Instr::Choice { alt: usize::MAX });
            let choice_site = out.len() - 1;
            let mut empty_site = None;
            if let Some(sep) = separator {
                out.push(Instr::IfEmpty { body: usize::MAX });
                empty_site = Some(out.len() - 1);
                emit(sep, atomic, index_of, out)?;
                if !atomic {
                    out.push(Instr::SkipWs);
                }
            }
            let body = out.len();
            if let Some(site) = empty_site {
                if let Instr::IfEmpty { body: target } = &mut out[site] {
                    *target = body;
                }
            }
            emit(pattern, atomic, index_of, out)?;
            out.push(Instr::RepeatEnd {
                body: top,
                exit: usize::MAX,
                max: *max,
            });
            let repeat_site = out.len() - 1;
            let exit = out.len();
            if let Instr::Choice { alt } = &mut out[choice_site] {
                *alt = exit;
            }
            if let Instr::RepeatEnd { exit: target, .. } = &mut out[repeat_site] {
                *target = exit;
            }
            out.push(Instr::ListEnd {
                min: *min,
                binding: direct_binding(pattern),
            });
            Ok(())
        }
        PatternIR::PositiveLookahead(inner) => {
            out.push(Instr::BeginLook {
                negative: false,
                end: usize::MAX,
            });
            let site = out.len() - 1;
            emit(inner, atomic, index_of, out)?;
            out.push(Instr::EndLook { negative: false });
            let here = out.len();
            if let Instr::BeginLook { end, .. } = &mut out[site] {
                *end = here;
            }
            Ok(())
        }
        PatternIR::NegativeLookahead(inner) => {
            out.push(Instr::BeginLook {
                negative: true,
                end: usize::MAX,
            });
            let site = out.len() - 1;
            emit(inner, atomic, index_of, out)?;
            out.push(Instr::EndLook { negative: true });
            let here = out.len();
            if let Instr::BeginLook { end, .. } = &mut out[site] {
                *end = here;
            }
            Ok(())
        }
    }
}

// =========================================================================
// Execution
// =========================================================================

/// Why a backtrack entry was pushed.
#[derive(Clone, Copy)]
enum CpKind {
    /// An alternative to try, or a repetition to leave.
    Alt,
    /// A lookahead in progress.
    Look { negative: bool },
}

/// A point to resume from when something fails.
struct Choicepoint {
    pc: usize,
    pos: usize,
    bindings: usize,
    lists: usize,
    kind: CpKind,
}

/// One rule in flight.
struct Frame {
    rule: usize,
    /// Where to resume in the caller, or `usize::MAX` for the rule the
    /// machine was entered on.
    return_pc: usize,
    start_pos: usize,
    /// The caller's bindings, restored when the rule leaves.
    bindings: usize,
    /// Stack depths on entry, so a failing rule leaves nothing behind.
    cp_floor: usize,
    list_floor: usize,
}

/// What entering a rule decided.
enum Enter {
    /// Run the body from this frame.
    Run(Frame),
    /// The memo already had a result.
    Value(ParsedValue),
    /// The memo already had a failure, or the rule is left-recursive.
    Fail,
}

/// Enter a rule: check the memo, mark it in progress, and open a
/// binding scope. This is the interpreter's rule prologue, so both
/// halves agree on what a rule costs and what it remembers.
fn enter<'g>(
    program: &Program<'g>,
    rule: usize,
    state: &mut ParserState<'_>,
    return_pc: usize,
    cp_floor: usize,
    list_floor: usize,
) -> Enter {
    let memo_id = program.rules[rule].memo_id;
    let start_pos = state.pos();

    if let Some(entry) = state.check_memo(memo_id) {
        return match entry.clone() {
            MemoEntry::Success { value, end_pos } => {
                state.set_pos(end_pos);
                Enter::Value(value)
            }
            MemoEntry::Failure => {
                state.set_pos(start_pos);
                let _: ParseResult<()> = state.fail("memoized failure");
                Enter::Fail
            }
            MemoEntry::InProgress => {
                state.set_pos(start_pos);
                let _: ParseResult<()> = state.fail("left recursion detected");
                Enter::Fail
            }
        };
    }
    state.store_memo(memo_id, MemoEntry::InProgress);

    // Each rule has its own binding scope, so an inner rule cannot
    // overwrite a name its caller is still holding.
    let bindings = state.save_bindings();
    state.clear_bindings();

    Enter::Run(Frame {
        rule,
        return_pc,
        start_pos,
        bindings,
        cp_floor,
        list_floor,
    })
}

/// Run a compiled rule.
///
/// The caller has already checked that `rule` has machine code; a rule
/// it calls that does not is run by the interpreter and its result
/// taken back here.
pub fn run<'g>(
    program: &Program<'g>,
    interp: &GrammarInterpreter<'g>,
    rule: usize,
    state: &mut ParserState<'_>,
) -> ParseResult<ParsedValue> {
    let mut frames: Vec<Frame> = Vec::with_capacity(64);
    let mut cps: Vec<Choicepoint> = Vec::with_capacity(64);
    let mut lists: Vec<Vec<ParsedValue>> = Vec::with_capacity(16);
    let mut v = ParsedValue::None;

    let mut pc = match enter(program, rule, state, usize::MAX, 0, 0) {
        Enter::Run(frame) => {
            let entry = program.rules[frame.rule].entry.expect("machine rule");
            frames.push(frame);
            entry
        }
        Enter::Value(value) => {
            let end = state.pos();
            return ParseResult::Success(value, end);
        }
        Enter::Fail => return ParseResult::Failure(state.furthest_error()),
    };

    'step: loop {
        let mut failed = false;

        match &program.code[pc] {
            Instr::Literal { text, desc } => {
                if state.check(text) {
                    for _ in text.chars() {
                        state.advance();
                    }
                    v = ParsedValue::None;
                    pc += 1;
                } else {
                    let _: ParseResult<()> = state.fail(desc);
                    failed = true;
                }
            }

            Instr::Class(class) => match interp.execute_char_class(class, state) {
                ParseResult::Success(value, _) => {
                    v = value;
                    pc += 1;
                }
                ParseResult::Failure(_) => failed = true,
            },

            Instr::AnyChar => match state.advance() {
                Some(c) => {
                    v = ParsedValue::Text(c.to_string());
                    pc += 1;
                }
                None => {
                    let _: ParseResult<()> = state.fail("any character");
                    failed = true;
                }
            },

            Instr::Soi => {
                if state.pos() == 0 {
                    v = ParsedValue::None;
                    pc += 1;
                } else {
                    let _: ParseResult<()> = state.fail("start of input");
                    failed = true;
                }
            }

            Instr::Eoi => {
                if state.is_eof() {
                    v = ParsedValue::None;
                    pc += 1;
                } else {
                    let _: ParseResult<()> = state.fail("end of input");
                    failed = true;
                }
            }

            Instr::SkipWs => {
                state.skip_ws();
                pc += 1;
            }

            Instr::SetNone => {
                v = ParsedValue::None;
                pc += 1;
            }

            Instr::Bind(name) => {
                state.set_binding(name, v.clone());
                pc += 1;
            }

            Instr::Fail => failed = true,

            Instr::Choice { alt } => {
                cps.push(Choicepoint {
                    pc: *alt,
                    pos: state.pos(),
                    bindings: state.save_bindings(),
                    lists: lists.len(),
                    kind: CpKind::Alt,
                });
                pc += 1;
            }

            Instr::Commit { next } => {
                cps.pop();
                pc = *next;
            }

            Instr::OptSome => {
                let inner = std::mem::replace(&mut v, ParsedValue::None);
                v = ParsedValue::Optional(Some(Box::new(inner)));
                pc += 1;
            }

            Instr::OptNone { binding } => {
                if let Some(name) = binding {
                    state.set_binding(name, ParsedValue::Optional(None));
                }
                v = ParsedValue::Optional(None);
                pc += 1;
            }

            Instr::ListBegin => {
                lists.push(Vec::new());
                pc += 1;
            }

            Instr::IfEmpty { body } => {
                if lists.last().is_some_and(|l| l.is_empty()) {
                    pc = *body;
                } else {
                    pc += 1;
                }
            }

            Instr::RepeatEnd { body, exit, max } => {
                let cp = cps.pop().expect("repetition choice point");
                if state.pos() == cp.pos {
                    // A pass that consumed nothing would repeat
                    // forever. Drop it and leave.
                    state.restore_bindings(cp.bindings);
                    pc = *exit;
                } else {
                    let item = std::mem::replace(&mut v, ParsedValue::None);
                    let list = lists.last_mut().expect("repetition list");
                    list.push(item);
                    pc = match max {
                        Some(m) if list.len() >= *m => *exit,
                        _ => *body,
                    };
                }
            }

            Instr::ListEnd { min, binding } => {
                let items = lists.pop().expect("repetition list");
                if items.len() < *min {
                    let _: ParseResult<()> = state.fail(&format!(
                        "expected at least {} items, got {}",
                        min,
                        items.len()
                    ));
                    failed = true;
                } else {
                    if let Some(name) = binding {
                        state.set_binding(name, ParsedValue::List(items.clone()));
                    }
                    v = ParsedValue::List(items);
                    pc += 1;
                }
            }

            Instr::BeginLook { negative, end } => {
                cps.push(Choicepoint {
                    pc: *end,
                    pos: state.pos(),
                    bindings: state.save_bindings(),
                    lists: lists.len(),
                    kind: CpKind::Look {
                        negative: *negative,
                    },
                });
                pc += 1;
            }

            Instr::EndLook { negative } => {
                // The inner pattern matched. Either way the lookahead
                // consumes nothing, so unwind to where it began.
                let floor = frames.last().map_or(0, |f| f.cp_floor);
                let mut resumed = None;
                while cps.len() > floor {
                    let cp = cps.pop().expect("lookahead choice point");
                    if matches!(cp.kind, CpKind::Look { .. }) {
                        resumed = Some(cp);
                        break;
                    }
                }
                let cp = resumed.expect("lookahead choice point");
                state.set_pos(cp.pos);
                state.restore_bindings(cp.bindings);
                lists.truncate(cp.lists);
                if *negative {
                    let _: ParseResult<()> = state.fail("negative lookahead matched");
                    failed = true;
                } else {
                    v = ParsedValue::None;
                    pc += 1;
                }
            }

            Instr::Call { rule: callee, .. } => {
                let callee = *callee;
                match program.rules[callee].entry {
                    Some(entry) => {
                        match enter(program, callee, state, pc + 1, cps.len(), lists.len()) {
                            Enter::Run(frame) => {
                                frames.push(frame);
                                pc = entry;
                            }
                            Enter::Value(value) => {
                                if let Instr::Call {
                                    binding: Some(name),
                                    ..
                                } = &program.code[pc]
                                {
                                    state.set_binding(name, value.clone());
                                }
                                v = value;
                                pc += 1;
                            }
                            Enter::Fail => failed = true,
                        }
                    }
                    // A form the machine does not express yet. The
                    // interpreter runs that rule and hands the value
                    // back, so coverage can grow one rule at a time.
                    None => {
                        let slot = &program.rules[callee];
                        match interp.execute_rule_with_id(slot.rule, slot.memo_id, state) {
                            ParseResult::Success(value, _) => {
                                if let Instr::Call {
                                    binding: Some(name),
                                    ..
                                } = &program.code[pc]
                                {
                                    state.set_binding(name, value.clone());
                                }
                                v = value;
                                pc += 1;
                            }
                            ParseResult::Failure(_) => failed = true,
                        }
                    }
                }
            }

            Instr::Ret => {
                let frame = frames.pop().expect("frame");
                let slot = &program.rules[frame.rule];
                let end_pos = state.pos();

                let mut value = std::mem::replace(&mut v, ParsedValue::None);
                if slot.atomic {
                    // An atomic rule's value is the text it matched,
                    // which `text()` reads back through this name.
                    let text = state.slice(frame.start_pos, end_pos).to_string();
                    state.set_binding("__text__", ParsedValue::Text(text.clone()));
                    value = ParsedValue::Text(text);
                }

                let produced = match &slot.rule.action {
                    Some(action) => {
                        let span = Span::new(frame.start_pos, end_pos);
                        match interp.execute_action(action, state, span) {
                            Ok(result) => Some(result),
                            Err(e) => {
                                let _: ParseResult<()> = state.fail(&e);
                                None
                            }
                        }
                    }
                    None => Some(value),
                };

                state.restore_bindings(frame.bindings);
                lists.truncate(frame.list_floor);
                cps.truncate(frame.cp_floor);

                match produced {
                    Some(result) => {
                        state.store_memo_at(
                            frame.start_pos,
                            slot.memo_id,
                            MemoEntry::Success {
                                value: result.clone(),
                                end_pos,
                            },
                        );
                        if frame.return_pc == usize::MAX {
                            return ParseResult::Success(result, end_pos);
                        }
                        if let Instr::Call {
                            binding: Some(name),
                            ..
                        } = &program.code[frame.return_pc - 1]
                        {
                            state.set_binding(name, result.clone());
                        }
                        v = result;
                        pc = frame.return_pc;
                    }
                    None => {
                        state.store_memo_at(frame.start_pos, slot.memo_id, MemoEntry::Failure);
                        if frames.is_empty() {
                            return ParseResult::Failure(state.furthest_error());
                        }
                        failed = true;
                    }
                }
            }
        }

        if !failed {
            continue;
        }

        // Unwind to the nearest point that can resume: a choice point
        // in this rule, or the rule itself, whose caller then fails in
        // turn.
        loop {
            let floor = frames.last().map_or(0, |f| f.cp_floor);
            if cps.len() > floor {
                let cp = cps.pop().expect("choice point");
                state.set_pos(cp.pos);
                state.restore_bindings(cp.bindings);
                lists.truncate(cp.lists);
                match cp.kind {
                    CpKind::Alt => {
                        pc = cp.pc;
                        continue 'step;
                    }
                    // The inner pattern failed, which is what a
                    // negative lookahead is looking for.
                    CpKind::Look { negative: true } => {
                        v = ParsedValue::None;
                        pc = cp.pc;
                        continue 'step;
                    }
                    CpKind::Look { negative: false } => continue,
                }
            }

            let frame = frames.pop().expect("frame");
            let slot = &program.rules[frame.rule];
            state.set_pos(frame.start_pos);
            state.restore_bindings(frame.bindings);
            lists.truncate(frame.list_floor);
            state.store_memo_at(frame.start_pos, slot.memo_id, MemoEntry::Failure);
            if frames.is_empty() {
                return ParseResult::Failure(state.furthest_error());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{GrammarIR, RuleIR};
    use zyntax_typed_ast::type_registry::TypeRegistry;
    use zyntax_typed_ast::TypedASTBuilder;

    fn grammar_with(rules: Vec<(&str, PatternIR)>) -> GrammarIR {
        let mut g = GrammarIR::default();
        for (name, pattern) in rules {
            g.rules.insert(
                name.to_string(),
                RuleIR {
                    name: name.to_string(),
                    pattern,
                    action: None,
                    modifier: None,
                    return_type: None,
                },
            );
        }
        g
    }

    fn ids(grammar: &GrammarIR) -> HashMap<String, usize> {
        grammar
            .rules
            .keys()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect()
    }

    fn compiled(grammar: &GrammarIR) -> Program<'_> {
        compile(grammar, &ids(grammar))
    }

    /// Run a rule on the machine and report what it produced and how
    /// far it got.
    fn parse(grammar: &GrammarIR, rule: &str, input: &str) -> Option<(ParsedValue, usize)> {
        let interp = GrammarInterpreter::new(grammar);
        let program = compiled(grammar);
        let index = program.rule_index(rule).expect("rule");
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new(input, &mut builder, &mut registry);
        match run(&program, &interp, index, &mut state) {
            ParseResult::Success(v, pos) => Some((v, pos)),
            ParseResult::Failure(_) => None,
        }
    }

    fn lit(s: &str) -> PatternIR {
        PatternIR::Literal(s.to_string())
    }

    #[test]
    fn a_sequence_emits_its_elements_in_order() {
        let g = grammar_with(vec![("r", PatternIR::Sequence(vec![lit("a"), lit("b")]))]);
        let p = compiled(&g);
        assert_eq!(p.unsupported.len(), 0, "a sequence of literals compiles");
        // literal, skip-ws, literal, ret
        assert_eq!(p.code.len(), 4);
        assert!(matches!(p.code[0], Instr::Literal { .. }));
        assert!(matches!(p.code[3], Instr::Ret));
    }

    #[test]
    fn a_choice_guards_every_alternative_but_the_last() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Choice(vec![lit("a"), lit("b"), lit("c")]),
        )]);
        let p = compiled(&g);
        let choices = p
            .code
            .iter()
            .filter(|i| matches!(i, Instr::Choice { .. }))
            .count();
        assert_eq!(choices, 2, "three alternatives need two choice points");
    }

    #[test]
    fn a_choice_point_resumes_at_the_next_alternative() {
        let g = grammar_with(vec![("r", PatternIR::Choice(vec![lit("a"), lit("b")]))]);
        let p = compiled(&g);
        let Instr::Choice { alt } = p.code[0] else {
            panic!("expected a choice point first, got {:?}", p.code[0]);
        };
        assert!(
            matches!(&p.code[alt], Instr::Literal { text, .. } if text == "b"),
            "the choice point lands on the second alternative, got {:?}",
            p.code[alt]
        );
    }

    #[test]
    fn a_repetition_loops_back_to_its_own_pass() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Repeat {
                pattern: Box::new(lit("a")),
                min: 0,
                max: None,
                separator: None,
            },
        )]);
        let p = compiled(&g);
        let Some(Instr::RepeatEnd { body, .. }) = p
            .code
            .iter()
            .find(|i| matches!(i, Instr::RepeatEnd { .. }))
            .cloned()
        else {
            panic!("expected a repetition end, got {:?}", p.code);
        };
        assert!(
            matches!(p.code[body], Instr::SkipWs),
            "a pass begins by skipping whitespace, got {:?}",
            p.code[body]
        );
    }

    #[test]
    fn a_rule_body_is_placed_at_its_own_entry() {
        // Two rules, so the second body's jumps have to be shifted.
        let g = grammar_with(vec![
            ("a", PatternIR::Choice(vec![lit("x"), lit("y")])),
            ("b", PatternIR::Choice(vec![lit("p"), lit("q")])),
        ]);
        let p = compiled(&g);
        let second = p.rules[1].entry.expect("second rule compiles");
        let Instr::Choice { alt } = p.code[second] else {
            panic!("expected a choice point, got {:?}", p.code[second]);
        };
        assert!(
            alt > second,
            "a jump target lands inside its own body, got {alt} for a body at {second}"
        );
        assert!(
            matches!(&p.code[alt], Instr::Literal { text, .. } if text == "q"),
            "got {:?}",
            p.code[alt]
        );
    }

    #[test]
    fn a_sequence_of_literals_consumes_them() {
        let g = grammar_with(vec![("r", PatternIR::Sequence(vec![lit("a"), lit("b")]))]);
        assert_eq!(parse(&g, "r", "a b").map(|(_, pos)| pos), Some(3));
        assert_eq!(parse(&g, "r", "a c").map(|(_, pos)| pos), None);
    }

    #[test]
    fn a_choice_falls_through_to_a_later_alternative() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Choice(vec![
                PatternIR::Sequence(vec![lit("a"), lit("b")]),
                PatternIR::Sequence(vec![lit("a"), lit("c")]),
            ]),
        )]);
        assert_eq!(
            parse(&g, "r", "a c").map(|(_, pos)| pos),
            Some(3),
            "the second alternative retries from where the first began"
        );
    }

    #[test]
    fn a_repetition_collects_what_it_matched() {
        let g = grammar_with(vec![
            (
                "r",
                PatternIR::Repeat {
                    pattern: Box::new(PatternIR::RuleRef {
                        rule_name: "item".to_string(),
                        binding: Some("i".to_string()),
                    }),
                    min: 0,
                    max: None,
                    separator: None,
                },
            ),
            ("item", lit("a")),
        ]);
        let (value, pos) = parse(&g, "r", "a a a").expect("parses");
        let ParsedValue::List(items) = value else {
            panic!("a repetition returns a list, got {value:?}");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(pos, 5);
    }

    #[test]
    fn a_repetition_below_its_minimum_fails() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Repeat {
                pattern: Box::new(lit("a")),
                min: 2,
                max: None,
                separator: None,
            },
        )]);
        assert!(parse(&g, "r", "a b").is_none(), "one item is not two");
        assert_eq!(parse(&g, "r", "a a").map(|(_, pos)| pos), Some(3));
    }

    #[test]
    fn a_repetition_stops_at_its_maximum() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Repeat {
                pattern: Box::new(lit("a")),
                min: 0,
                max: Some(2),
                separator: None,
            },
        )]);
        let (value, pos) = parse(&g, "r", "aaa").expect("parses");
        let ParsedValue::List(items) = value else {
            panic!("expected a list, got {value:?}");
        };
        assert_eq!(items.len(), 2, "the third item is left for the caller");
        assert_eq!(pos, 2);
    }

    #[test]
    fn a_separated_repetition_runs_its_separator_between_items() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Repeat {
                pattern: Box::new(lit("a")),
                min: 0,
                max: None,
                separator: Some(Box::new(lit(","))),
            },
        )]);
        let (value, pos) = parse(&g, "r", "a, a, a").expect("parses");
        let ParsedValue::List(items) = value else {
            panic!("expected a list, got {value:?}");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(pos, 7);
    }

    #[test]
    fn an_absent_optional_consumes_nothing() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Sequence(vec![PatternIR::Optional(Box::new(lit("a"))), lit("b")]),
        )]);
        let (_, pos) = parse(&g, "r", "b").expect("parses without the optional");
        assert_eq!(pos, 1);
        assert_eq!(parse(&g, "r", "a b").map(|(_, p)| p), Some(3));
    }

    #[test]
    fn an_optional_reports_whether_it_matched() {
        let g = grammar_with(vec![("r", PatternIR::Optional(Box::new(lit("a"))))]);
        assert!(matches!(
            parse(&g, "r", "a"),
            Some((ParsedValue::Optional(Some(_)), 1))
        ));
        assert!(matches!(
            parse(&g, "r", "z"),
            Some((ParsedValue::Optional(None), 0))
        ));
    }

    #[test]
    fn a_negative_lookahead_succeeds_on_what_it_rejects() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Sequence(vec![
                PatternIR::NegativeLookahead(Box::new(lit("a"))),
                lit("b"),
            ]),
        )]);
        assert_eq!(parse(&g, "r", "b").map(|(_, pos)| pos), Some(1));
        assert!(parse(&g, "r", "a").is_none());
    }

    #[test]
    fn a_positive_lookahead_consumes_nothing() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Sequence(vec![
                PatternIR::PositiveLookahead(Box::new(lit("a"))),
                lit("a"),
            ]),
        )]);
        assert_eq!(parse(&g, "r", "a").map(|(_, pos)| pos), Some(1));
    }

    #[test]
    fn a_rule_the_machine_cannot_run_falls_back_to_the_interpreter() {
        let g = grammar_with(vec![
            (
                "r",
                PatternIR::Sequence(vec![
                    PatternIR::RuleRef {
                        rule_name: "inner".to_string(),
                        binding: None,
                    },
                    lit("!"),
                ]),
            ),
            ("inner", lit("hi")),
        ]);
        let interp = GrammarInterpreter::new(&g);
        let mut program = compiled(&g);
        // Take away the callee's code, leaving the caller to reach it
        // the way it reaches a form the compiler does not cover.
        let inner = program.rule_index("inner").expect("rule");
        program.rules[inner].entry = None;

        let index = program.rule_index("r").expect("rule");
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("hi!", &mut builder, &mut registry);
        let result = run(&program, &interp, index, &mut state);
        assert!(
            matches!(result, ParseResult::Success(_, 3)),
            "the interpreter ran the callee, got {result:?}"
        );
    }

    #[test]
    fn the_machine_and_the_interpreter_agree() {
        let source = r#"
            @language { name: "MachineTest", version: "1.0" }

            program = { SOI ~ items:item* ~ EOI }
            item = { list | word }
            list = { "[" ~ items:word* ~ "]" }
            word = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
            WHITESPACE = _{ " " | "\t" | "\n" }
        "#;
        let grammar = crate::grammar::parse_grammar(source).expect("grammar parses");
        let interp = GrammarInterpreter::new(&grammar);
        let program = compiled(&grammar);
        assert_eq!(program.unsupported.len(), 0, "every rule compiles");

        for input in ["a b c", "[x y] z", "[]", "a [b] c d", "", "a ["] {
            let mut b1 = TypedASTBuilder::new();
            let mut r1 = TypeRegistry::new();
            let mut s1 = ParserState::new(input, &mut b1, &mut r1);
            let expected = interp.parse_rule("program", &mut s1);

            let mut b2 = TypedASTBuilder::new();
            let mut r2 = TypeRegistry::new();
            let mut s2 = ParserState::new(input, &mut b2, &mut r2);
            let index = program.rule_index("program").expect("rule");
            let actual = run(&program, &interp, index, &mut s2);

            match (&expected, &actual) {
                (ParseResult::Success(a, p), ParseResult::Success(b, q)) => {
                    assert_eq!(p, q, "same end position on {input:?}");
                    assert_eq!(
                        format!("{a:?}"),
                        format!("{b:?}"),
                        "same value on {input:?}"
                    );
                }
                (ParseResult::Failure(_), ParseResult::Failure(_)) => {}
                _ => panic!(
                    "machine and interpreter disagree on {input:?}: {expected:?} vs {actual:?}"
                ),
            }
        }
    }
}
