function apply(f: (n: number) => number, n: number): number { return f(n); }
let f = (n: number) => n + 1;
const original = f;
f = (n: number) => n * 2;
console.log(original(3), f(3), apply(original, 5));
