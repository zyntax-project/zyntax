use crate::context::{LoweringTarget, MatchContext};
use crate::metadata::MetadataTable;
use crate::node_id::NodeId;
use crate::pass::PatternPass;
use crate::rewrite::{DeclRewrite, ExprRewrite, StmtRewrite};
use crate::trace::{FiredRewrite, FiredSet};
use crate::walk;
use zyntax_typed_ast::advanced_analysis::AnalysisContext;
use zyntax_typed_ast::diagnostics::Diagnostic;
use zyntax_typed_ast::effect_system::EffectSystem;
use zyntax_typed_ast::typed_ast::TypedProgram;
use zyntax_typed_ast::{TypeRegistry, TypedASTBuilder};

/// Configuration for the pattern engine.
pub struct EngineConfig {
    /// Maximum fixpoint iterations before giving up.
    pub max_iterations: usize,
    /// Current lowering target.
    pub target: LoweringTarget,
    /// Emit rewrite trace for debugging.
    pub trace: bool,
    /// Run type checker after each iteration (debug builds).
    pub verify_after: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            target: LoweringTarget::Cpu,
            trace: false,
            verify_after: false,
        }
    }
}

/// Result of running the engine.
pub struct EngineResult {
    /// Number of fixpoint iterations completed.
    pub iterations: usize,
    /// All rewrites that fired, in order.
    pub rewrites_fired: Vec<FiredRewrite>,
    /// Whether the program was modified.
    pub changed: bool,
    /// Diagnostics generated during engine execution.
    pub diagnostics: Vec<Diagnostic>,
}

/// Error from pass ordering.
#[derive(Debug)]
pub enum PassOrderError {
    /// A pass depends on another that wasn't registered.
    MissingDependency {
        pass: &'static str,
        dependency: &'static str,
    },
    /// A cycle was detected in pass dependencies.
    Cycle(Vec<&'static str>),
}

impl std::fmt::Display for PassOrderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassOrderError::MissingDependency { pass, dependency } => {
                write!(
                    f,
                    "pass '{}' depends on '{}' which is not registered",
                    pass, dependency
                )
            }
            PassOrderError::Cycle(names) => {
                write!(f, "cycle in pass dependencies: {:?}", names)
            }
        }
    }
}

impl std::error::Error for PassOrderError {}

/// The pattern engine. Holds registered rewrites and passes.
pub struct PatternEngine {
    expr_rewrites: Vec<ExprRewrite>,
    stmt_rewrites: Vec<StmtRewrite>,
    decl_rewrites: Vec<DeclRewrite>,
    passes: Vec<Box<dyn PatternPass>>,
    pass_order: Vec<&'static str>,
    config: EngineConfig,
    pub metadata: MetadataTable,
}

impl PatternEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            expr_rewrites: Vec::new(),
            stmt_rewrites: Vec::new(),
            decl_rewrites: Vec::new(),
            passes: Vec::new(),
            pass_order: Vec::new(),
            config,
            metadata: MetadataTable::new(),
        }
    }

    // --- Registration ---

    pub fn register_expr_rewrite(&mut self, r: ExprRewrite) -> &mut Self {
        self.expr_rewrites.push(r);
        self
    }

    pub fn register_stmt_rewrite(&mut self, r: StmtRewrite) -> &mut Self {
        self.stmt_rewrites.push(r);
        self
    }

    pub fn register_decl_rewrite(&mut self, r: DeclRewrite) -> &mut Self {
        self.decl_rewrites.push(r);
        self
    }

    pub fn register_pass(&mut self, pass: impl PatternPass + 'static) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Resolve pass dependencies and sort passes into execution order.
    /// Then let each pass register its rewrites.
    pub fn finalize(&mut self) -> Result<(), PassOrderError> {
        // Topological sort of passes by dependencies
        let pass_names: Vec<&'static str> = self.passes.iter().map(|p| p.name()).collect();

        // Check for missing dependencies
        for pass in &self.passes {
            for dep in pass.dependencies() {
                if !pass_names.contains(dep) {
                    return Err(PassOrderError::MissingDependency {
                        pass: pass.name(),
                        dependency: dep,
                    });
                }
            }
        }

        // Simple topological sort (Kahn's algorithm)
        let mut in_degree: std::collections::HashMap<&'static str, usize> =
            pass_names.iter().map(|&n| (n, 0)).collect();
        let mut dependents: std::collections::HashMap<&'static str, Vec<&'static str>> =
            std::collections::HashMap::new();

        for pass in &self.passes {
            for dep in pass.dependencies() {
                *in_degree.entry(pass.name()).or_default() += 1;
                dependents.entry(dep).or_default().push(pass.name());
            }
        }

        let mut queue: std::collections::VecDeque<&'static str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&name, _)| name)
            .collect();

        let mut sorted = Vec::new();
        while let Some(name) = queue.pop_front() {
            sorted.push(name);
            if let Some(deps) = dependents.get(name) {
                for &dep in deps {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
        }

        if sorted.len() != pass_names.len() {
            let remaining: Vec<&'static str> = pass_names
                .into_iter()
                .filter(|n| !sorted.contains(n))
                .collect();
            return Err(PassOrderError::Cycle(remaining));
        }

        self.pass_order = sorted;

        // Let each pass register its rewrites (in order).
        // Take passes out to avoid borrow conflict with `self`.
        let passes = std::mem::take(&mut self.passes);
        let pass_order = self.pass_order.clone();
        for name in &pass_order {
            if let Some(pass) = passes.iter().find(|p| p.name() == *name) {
                pass.register(self);
            }
        }
        self.passes = passes;

        // Sort rewrites by priority (lower number = higher priority = fires first)
        self.expr_rewrites.sort_by_key(|r| r.priority);
        self.stmt_rewrites.sort_by_key(|r| r.priority);
        self.decl_rewrites.sort_by_key(|r| r.priority);

        Ok(())
    }

    // --- Execution ---

    /// Run the engine on a program to fixpoint.
    pub fn run(&mut self, program: &mut TypedProgram, registry: &TypeRegistry) -> EngineResult {
        NodeId::reset();

        let mut all_fired: Vec<FiredRewrite> = Vec::new();
        let mut any_changed = false;
        let mut iteration = 0;

        // Build analysis and effects (can be empty if not needed by any pattern)
        let analysis = AnalysisContext::new();
        let effects = EffectSystem::new();
        let mut builder = TypedASTBuilder::new();

        loop {
            iteration += 1;
            if iteration > self.config.max_iterations {
                log::warn!(
                    "Pattern engine hit max iterations ({}), stopping",
                    self.config.max_iterations
                );
                break;
            }

            let fired_set = FiredSet::new();
            let effect_ops = crate::context::EffectOpIndex::build(program);
            let ctx = MatchContext {
                registry,
                analysis: &analysis,
                effects: &effects,
                target: self.config.target,
                metadata: &self.metadata,
                fired: &fired_set,
                effect_ops: &effect_ops,
            };

            let walk_result = walk::walk_program(
                program,
                &self.expr_rewrites,
                &self.stmt_rewrites,
                &self.decl_rewrites,
                &ctx,
                &mut builder,
                iteration,
            );

            if self.config.trace && !walk_result.fired.is_empty() {
                for f in &walk_result.fired {
                    log::info!(
                        "[pattern_engine] iteration {} fired '{}' at {:?}",
                        f.iteration,
                        f.rewrite_name,
                        f.node_span,
                    );
                }
            }

            all_fired.extend(walk_result.fired);

            if !walk_result.changed {
                break; // fixpoint reached
            }

            any_changed = true;

            // Post-rewrite verification
            if self.config.verify_after {
                let errors = crate::verify::verify_type_soundness(program, registry);
                if !errors.is_empty() {
                    log::error!(
                        "[pattern_engine] type errors after iteration {}: {:?}",
                        iteration,
                        errors
                    );
                }
            }
        }

        EngineResult {
            iterations: iteration,
            rewrites_fired: all_fired,
            changed: any_changed,
            diagnostics: Vec::new(),
        }
    }
}
