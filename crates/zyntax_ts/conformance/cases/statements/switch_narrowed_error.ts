enum E { A, B }
function f(e: E): number {
    switch (e) {
        case E.A: return e === E.B ? 1 : 2;
        default: return 0;
    }
}
