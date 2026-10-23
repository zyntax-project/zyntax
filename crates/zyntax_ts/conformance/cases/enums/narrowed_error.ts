enum E { A, B, C }
function f(e: E): boolean { if (e === E.A) { return e === E.B; } return false; }
