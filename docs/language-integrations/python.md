# Python in zypy

## Integers

By default an `int` is 64 bits wide, and a result that does not fit wraps
around. A literal wider than 64 bits is refused. This keeps loops and index
arithmetic as fast as the machine allows.

If a program needs Python's unbounded integers, ask for them in one of two
ways. Put the comment `# zypy: bigint` before the first statement. Or run the
program with the environment variable `ZYPY_BIGINT=1` set. CPython ignores the
comment, so the same file runs on both.

With big integers on:

- If the result of `+`, `-`, `*`, `**` or `<<` on ints does not fit 64 bits,
  the result is a wider integer. A result that fits is the ordinary machine
  int, so a wide integer never equals a narrow one.
- Integer literals of any size work.
- `int(...)`, `abs`, `sum`, `pow` (also with a modulus), `divmod` and
  `isinstance(x, int)` accept wide integers.
- `math.factorial`, `gcd`, `lcm`, `isqrt`, `comb` and `prod` accept them too.
- Lists, tuples, dicts and sets hold ints as dynamic values, so any element can
  be a wide integer.

Wide integers cost time. Each `+`, `-`, `*` and `**` is a call that checks for
overflow, and the containers of ints are no longer typed. Use the switch for
programs whose answers need it.

What does not work with a wide integer:

- Format specs that change digits, such as `f"{n:,}"`, `f"{n:x}"` and
  `f"{n:030}"`. The value prints in decimal, and only alignment and width apply.
- `%d` formatting, `bin`, `hex` and `oct`.
- `range` and indexing with a value that does not fit 64 bits.
- Augmented assignment to a computed target, such as `xs[f(i)] += n`.
- Negating the smallest 64-bit integer.

The wide integer type is written in Python, in
`crates/zyntax_python/stdlib/bigint.py`, and compiled with the program.
