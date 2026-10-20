function tail(n: number) { if (n <= 0) return base(n); return tail(n - 1); }
function base(n: number) { return n + 4; }
console.log(tail(5));
