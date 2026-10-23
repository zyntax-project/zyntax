enum Op { Add, Sub, Mul }
function apply(op: Op, a: number, b: number): number {
    switch (op) {
        case Op.Add: return a + b;
        case Op.Sub: return a - b;
        case Op.Mul: return a * b;
    }
}
let ops = 0;
for (let i = 0; i < 3; i++) {
    switch (i) {
        case 0: ops += apply(Op.Add, 2, 3); continue;
        case 1: ops += apply(Op.Sub, 9, 4); break;
        default: ops += apply(Op.Mul, 2, 2);
    }
    ops += 100;
}
console.log(apply(Op.Mul, 6, 7), ops);
