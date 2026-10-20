type Transform = (n: number) => number;
function apply(f: Transform, value: number): number { return f(value); }
function select(): Transform { return n => n + 0.5; }
let f: Transform = n => n * 2;
f = n => n + 1;
console.log(f(3), apply(n => n * 3, 4), select()(1));
const asserted = ((n) => n + 2) as Transform;
console.log(asserted(5));
