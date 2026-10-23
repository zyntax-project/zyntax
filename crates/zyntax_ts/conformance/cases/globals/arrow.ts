let scale = 3;
const offset = 0.5;
const apply = (x: number): number => x * scale + offset;
function twice(f: (x: number) => number, x: number): number { return f(f(x)); }
console.log(apply(2), twice(apply, 1));
scale = 10;
console.log(apply(2));
