enum E { A, B, C }
function f(e: E): number {
    switch (e) {
        case E.A: return 1;
        case E.B: return 2;
    }
}
