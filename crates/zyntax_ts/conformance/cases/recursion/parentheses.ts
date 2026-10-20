function tail(n: number) { if (n <= 0) return 3; return ((tail(n - 1))); }
console.log(tail(5));
