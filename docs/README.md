# Zyntax Documentation

This directory contains architectural and educational documentation for the Zyntax compiler project.

## Documentation Structure

### Core Architecture
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Overall compiler architecture and design
- **[BYTECODE_FORMAT_SPEC.md](BYTECODE_FORMAT_SPEC.md)** - Bytecode format specification
- **[ASYNC_RUNTIME_DESIGN.md](ASYNC_RUNTIME_DESIGN.md)** - Async/await runtime design
- **[PLUGIN_ARCHITECTURE.md](PLUGIN_ARCHITECTURE.md)** - Plugin system architecture
- **[tiered-compilation.md](tiered-compilation.md)** - Tiered compilation strategy
- **[EMBEDDING_SDK.md](EMBEDDING_SDK.md)** - Embedding Zyntax in Rust applications (zyntax_embed)

### Implementation Guides
- **[HIR_BUILDER_EXAMPLE.md](HIR_BUILDER_EXAMPLE.md)** - HIR (High-level Intermediate Representation) builder examples
- **[PATTERN_MATCHING.md](PATTERN_MATCHING.md)** - Pattern matching implementation and architecture
- **[REFERENCE_TYPES.md](REFERENCE_TYPES.md)** - Value-type vs `@reference` class semantics, the bi-modal memory model, and lowering

### Language Integrations
Located in [language-integrations/](language-integrations/) directory:
- **Haxe Integration** - Haxe language integration documentation
- **Zyn Parser** - Zig language parser implementation

## Other Documentation

### Project Root
- **[../Readme.md](../Readme.md)** - Main project README
- **[../BACKLOG.md](../BACKLOG.md)** - Development backlog and task tracking

### Session Archives
Session-specific documentation is archived in `session-archive/` (untracked by git).
These documents capture implementation progress and decisions from specific development sessions
but are not intended as permanent reference material.

## Contributing to Documentation

When adding new documentation:
- **Architecture/Design docs**: Place in `docs/` root
- **Language integration docs**: Place in `docs/language-integrations/`
- **Session-specific status**: Place in `docs/session-archive/` (will not be committed)
- **Examples/Tutorials**: Place in `docs/` with descriptive names

Keep documentation:
- **Educational**: Focus on "why" not just "what"
- **Current**: Update when designs change
- **Accessible**: Use clear language and examples
- **Organized**: Follow the structure above
