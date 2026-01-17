# Changelog

## [0.2.0] - 2026-01-17

### Changed

- Removed `pub` keyword (all items are public by default)
- Added Python-style private item convention (`_` prefix)
- Private functions, methods, and variables now have distinct scopes:
  - `entity.name.function.private.zynml` for private function definitions
  - `entity.name.function.call.private.zynml` for private function calls
  - `entity.name.function.method.private.zynml` for private method calls
  - `variable.other.private.zynml` for private variables
  - `variable.parameter.private.zynml` for private parameters

## [0.1.0] - 2026-01-17

### Added

- Initial release of ZynML syntax highlighting
- Support for `.zynml` and `.zyn` file extensions
- Syntax highlighting for:
  - Keywords: `def`, `fn`, `struct`, `enum`, `trait`, `impl`, `type`, `let`, `mut`
  - Control flow: `if`, `else`, `elif`, `match`, `case`, `for`, `while`, `return`
  - Async: `async`, `await`
  - Effects: `effect`, `handler`, `where`
  - Primitive types: `i64`, `f64`, `bool`, `str`, `String`
  - Built-in types: `Option`, `Result`, `List`, `HashMap`, `Tensor`, `Iterator`
  - Operators: `|>`, `=>`, `->`, `??`, `?`, `::`
  - Annotations: `@effect`, `@device`, `@jit`
  - String literals and f-strings with interpolation
  - Numbers with suffixes (duration literals, etc.)
  - Comments (line and block)
- Language configuration for:
  - Bracket matching
  - Auto-closing pairs
  - Comment toggling
  - Indentation rules
