let calls = 0;
function next(): number { calls++; return calls; }
function f(a: number, b: number = a * 2, c?: number): number {
    return a + b + (c === undefined ? 100 : c);
}
function g(x: number = next(), y: boolean = true): string { return x + ":" + y; }
function sum(first: number, ...rest: number[]): number { let s = first; for (const r of rest) s += r; return s; }
console.log(f(1), f(1, 5), f(1, undefined, 3), f(2, 2, 2));
console.log(g(), g(10), g(undefined, false), calls);
console.log(sum(1), sum(1, 2, 3), sum(4, 5));
