function repeat<T>(x: T, n: number): T { if (n <= 0) return x; return repeat<T>(x, n - 1); }
console.log(repeat(7, 20), repeat("done", 10), repeat(false, 3));
