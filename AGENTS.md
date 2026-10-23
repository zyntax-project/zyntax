# TypeScript conformance

For changes to `zyntax_ts`, its conformance harness, or shared compiler/type-checking
code that can affect TypeScript, run the complete pinned official corpus before
committing. Local fixtures and focused upstream subsets do not replace this run.
Use the regression-baseline command in
[TypeScript testing](docs/language-integrations/typescript/testing.md).

Inspect per-case regressions and record outstanding findings in git-bug. Do not
refresh the reviewed baseline to hide lost passes, reduced coverage or failures.
Keep unsupported features and skipped cases distinct from compatibility passes.
