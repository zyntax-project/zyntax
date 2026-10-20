function f(f: (n: number) => number) { return f(2); }
console.log(f(n => n + 1));
