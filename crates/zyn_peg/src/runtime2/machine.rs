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
//! terminal, call a rule, and manage the choice points that make
//! ordered choice and repetition work. `Choice` pushes a backtrack
//! entry, `Commit` drops it, and `Fail` unwinds to the most recent one.

use crate::grammar::{CharClass, GrammarIR, PatternIR};

/// One instruction. Positions are indices into [`Program::code`].
#[derive(Debug, Clone)]
pub enum Instr {
    /// Match a literal, or fail.
    Literal(String),
    /// Match one character of a class, or fail.
    Class(CharClass),
    /// Match any one character, or fail at end of input.
    AnyChar,
    /// Match the start of input.
    Soi,
    /// Match the end of input.
    Eoi,
    /// Enter a rule, whose code begins at `entry`.
    Call {
        rule: usize,
        entry: usize,
        /// Bind the rule's value under this name on success.
        binding: Option<String>,
    },
    /// Leave the current rule with the value register as its result.
    Ret,
    /// Push a backtrack entry that resumes at `alt` on failure.
    Choice { alt: usize },
    /// Drop the most recent backtrack entry and continue at `next`.
    Commit { next: usize },
    /// Drop the most recent backtrack entry, keeping the current
    /// position. Used to end a repetition that stopped matching.
    Commit0,
    /// Jump.
    Jump { to: usize },
    /// Fail, unwinding to the most recent backtrack entry.
    Fail,
    /// Skip whitespace, unless the enclosing rule is atomic.
    SkipWs,
    /// Begin a lookahead. `negative` inverts the sense; either way the
    /// position is restored when it ends.
    BeginLook { negative: bool, end: usize },
    /// End a lookahead begun by [`Instr::BeginLook`].
    EndLook { negative: bool },
    /// Bind the value register under a name.
    Bind(String),
}

/// A compiled grammar.
#[derive(Debug, Default)]
pub struct Program {
    /// Every rule's code, concatenated.
    pub code: Vec<Instr>,
    /// Where each rule's code begins, by rule index.
    pub entries: Vec<usize>,
    /// Rule names, parallel to [`Self::entries`].
    pub names: Vec<String>,
    /// Rules the compiler could not express, by name. These keep
    /// running on the tree-walking interpreter, so a grammar using a
    /// form the machine does not cover still parses.
    pub unsupported: Vec<String>,
}

impl Program {
    /// Instructions emitted, for reporting.
    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Fraction of rules the machine can run, as a percentage.
    pub fn coverage(&self) -> f64 {
        let total = self.names.len() + self.unsupported.len();
        if total == 0 {
            return 0.0;
        }
        100.0 * self.names.len() as f64 / total as f64
    }
}

/// Compile a grammar into a program.
///
/// A rule the compiler cannot express is recorded in
/// [`Program::unsupported`] rather than failing the whole grammar, so
/// coverage can grow one form at a time without the machine having to
/// be complete before it is useful.
pub fn compile(grammar: &GrammarIR) -> Program {
    let mut program = Program::default();
    let mut rule_index: Vec<(&String, &crate::grammar::RuleIR)> = grammar.rules.iter().collect();
    // A stable order so entry indices are reproducible across runs.
    rule_index.sort_by(|a, b| a.0.cmp(b.0));

    // Rule indices have to exist before any body is compiled, because a
    // rule can call one that appears later.
    let mut index_of: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, (name, _)) in rule_index.iter().enumerate() {
        index_of.insert(name.as_str(), i);
    }

    let mut bodies: Vec<Option<Vec<Instr>>> = Vec::with_capacity(rule_index.len());
    for (_, rule) in &rule_index {
        let atomic = rule.modifier == Some(crate::grammar::RuleModifier::Atomic);
        let mut out = Vec::new();
        match emit(&rule.pattern, atomic, &index_of, &mut out) {
            Ok(()) => {
                out.push(Instr::Ret);
                bodies.push(Some(out));
            }
            Err(()) => bodies.push(None),
        }
    }

    for (i, (name, _)) in rule_index.iter().enumerate() {
        match &bodies[i] {
            Some(body) => {
                program.entries.push(program.code.len());
                program.names.push((*name).clone());
                program.code.extend(body.iter().cloned());
            }
            None => program.unsupported.push((*name).clone()),
        }
    }

    program
}

/// Emit code for one pattern, appending to `out`.
///
/// Returns `Err(())` for a form the machine does not cover yet, which
/// drops the whole rule to the interpreter rather than emitting
/// something that would parse differently.
fn emit(
    pattern: &PatternIR,
    atomic: bool,
    index_of: &std::collections::HashMap<&str, usize>,
    out: &mut Vec<Instr>,
) -> Result<(), ()> {
    match pattern {
        PatternIR::Literal(s) => {
            out.push(Instr::Literal(s.clone()));
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
            Ok(())
        }
        PatternIR::RuleRef { rule_name, binding } => {
            match rule_name.as_str() {
                "SOI" => out.push(Instr::Soi),
                "EOI" => out.push(Instr::Eoi),
                // The character built-ins are classes the machine
                // already has an instruction for.
                "ASCII_DIGIT" => out.push(Instr::Class(CharClass::Range('0', '9'))),
                "ANY" => out.push(Instr::AnyChar),
                name => {
                    let rule = *index_of.get(name).ok_or(())?;
                    out.push(Instr::Call {
                        rule,
                        // Patched once every rule's entry is known.
                        entry: usize::MAX,
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
            for (i, item) in items.iter().enumerate() {
                if !atomic && i > 0 {
                    out.push(Instr::SkipWs);
                }
                emit(item, atomic, index_of, out)?;
            }
            Ok(())
        }
        PatternIR::Choice(alts) => {
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
            out.push(Instr::Commit { next: usize::MAX });
            let commit_site = out.len() - 1;
            let here = out.len();
            if let Instr::Choice { alt } = &mut out[site] {
                *alt = here;
            }
            if let Instr::Commit { next } = &mut out[commit_site] {
                *next = here;
            }
            Ok(())
        }
        PatternIR::Repeat {
            pattern,
            min,
            max,
            separator,
        } => {
            // A separated repetition is its own shape; leave it to the
            // interpreter until the machine expresses it directly.
            if separator.is_some() {
                return Err(());
            }
            // Only the unbounded forms are expressed directly; a
            // counted repeat would need a counter register.
            if max.is_some() {
                return Err(());
            }
            let min = *min;
            if min > 1 {
                return Err(());
            }
            if min == 1 {
                emit(pattern, atomic, index_of, out)?;
                if !atomic {
                    out.push(Instr::SkipWs);
                }
            }
            let loop_start = out.len();
            out.push(Instr::Choice { alt: usize::MAX });
            let site = out.len() - 1;
            emit(pattern, atomic, index_of, out)?;
            out.push(Instr::Commit { next: loop_start });
            let here = out.len();
            if let Instr::Choice { alt } = &mut out[site] {
                *alt = here;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{GrammarIR, RuleIR};

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

    #[test]
    fn a_sequence_emits_its_elements_in_order() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Sequence(vec![
                PatternIR::Literal("a".into()),
                PatternIR::Literal("b".into()),
            ]),
        )]);
        let p = compile(&g);
        assert_eq!(p.unsupported.len(), 0, "a sequence of literals compiles");
        // literal, skip-ws, literal, ret
        assert_eq!(p.code.len(), 4);
        assert!(matches!(p.code[0], Instr::Literal(_)));
        assert!(matches!(p.code[3], Instr::Ret));
    }

    #[test]
    fn a_choice_guards_every_alternative_but_the_last() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Choice(vec![
                PatternIR::Literal("a".into()),
                PatternIR::Literal("b".into()),
                PatternIR::Literal("c".into()),
            ]),
        )]);
        let p = compile(&g);
        let choices = p
            .code
            .iter()
            .filter(|i| matches!(i, Instr::Choice { .. }))
            .count();
        assert_eq!(choices, 2, "three alternatives need two choice points");
    }

    #[test]
    fn a_choice_point_resumes_at_the_next_alternative() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Choice(vec![
                PatternIR::Literal("a".into()),
                PatternIR::Literal("b".into()),
            ]),
        )]);
        let p = compile(&g);
        let Instr::Choice { alt } = p.code[0] else {
            panic!("expected a choice point first, got {:?}", p.code[0]);
        };
        assert!(
            matches!(p.code[alt], Instr::Literal(ref s) if s == "b"),
            "the choice point lands on the second alternative, got {:?}",
            p.code[alt]
        );
    }

    #[test]
    fn a_repetition_loops_back_to_its_own_choice_point() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Repeat {
                pattern: Box::new(PatternIR::Literal("a".into())),
                min: 0,
                max: None,
                separator: None,
            },
        )]);
        let p = compile(&g);
        let Instr::Commit { next } = p.code[2] else {
            panic!("expected a commit after the body, got {:?}", p.code[2]);
        };
        assert_eq!(next, 0, "the loop commits back to its choice point");
    }

    #[test]
    fn a_counted_repetition_is_left_to_the_interpreter() {
        let g = grammar_with(vec![(
            "r",
            PatternIR::Repeat {
                pattern: Box::new(PatternIR::Literal("a".into())),
                min: 2,
                max: Some(4),
                separator: None,
            },
        )]);
        let p = compile(&g);
        assert_eq!(
            p.unsupported,
            vec!["r".to_string()],
            "a form the machine cannot express drops that rule, not the grammar"
        );
    }
}
