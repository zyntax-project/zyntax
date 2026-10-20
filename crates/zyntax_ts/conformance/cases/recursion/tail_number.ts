function before(n: number) { if (n <= 0) return 1.5; return before(n - 1); }
function after(n: number) { if (n > 0) return after(n - 1); return 2.5; }
console.log(before(5), after(5));
