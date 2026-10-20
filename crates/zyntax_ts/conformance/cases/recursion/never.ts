function forever(n: number) { return forever(n); }
function delegated(n: number) { return forever(n); }
function safe(n: number) { if (n < 0) return forever(n); return 6.5; }
console.log(safe(1));
