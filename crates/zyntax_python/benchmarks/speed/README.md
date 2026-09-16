# The PyPy speed-center kernels

The pure-Python kernels behind speed.pypy.org's "How fast is PyPy?"
chart, as the PyPy benchmarks repository carries them (the revision and
the files are in `kernels/REVISION`; `kernels/LICENSE` is theirs), run
on zypy, PyPy and CPython the way the speed center runs them. Each row
first says what matters: whether zypy is faster or slower than the
fastest of the other runtimes on that kernel, and by how much. Beside
it is the speed center's own figure, each runtime's mean iteration
time relative to CPython (lower is faster); the whole process's wall
time is in the JSON.

    python3 run.py            # 50 iterations times the speed center's scaling
    python3 run.py --fast     # 5 iterations, a quick look
    python3 run.py --only richards,nbody_modified

Build the `zypy` used by this harness with
`cargo build --release -p zyntax_python --features llvm-backend`.
The harness selects LLVM for `zypy` and checks that the binary provides it.

`run.py` looks for `target/release/zypy`, `pypy3` and `python3`;
`--zypy`, `--pypy` and `--python` name them. Results go to `speed.json`.

A kernel is never edited to fit. `shims/util.py` is the speed center's
own helper written out (theirs execs a file from another directory);
`shims/optparse.py` stands in for the module on zypy alone. A kernel an
interpreter cannot run is a row that says so, with the diagnostic.
