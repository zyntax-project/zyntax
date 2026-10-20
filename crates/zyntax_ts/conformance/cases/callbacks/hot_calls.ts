type Transform = (n: number) => number;
function apply(f: Transform, n: number): number { return f(n); }
function choose(): Transform { return n => n + 0.25; }
const f = choose();
let sum = 0;
for (let i = 0; i < 500; i += 1) sum += apply(f, i);
console.log(sum);
