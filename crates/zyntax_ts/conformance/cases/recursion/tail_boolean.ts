function flag(n: number, b: boolean) { if (n <= 0) return b; return flag(n - 1, b); }
console.log(flag(4, true), flag(4, false));
