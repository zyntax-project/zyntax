type Point = { x: number; y: number };
function make(n: number) { if (n <= 0) return { x: 3, y: 4 }; return make(n - 1); }
function alias(n: number, p: Point) { if (n <= 0) return p; return alias(n - 1, p); }
const p = make(5); const q = alias(5, p); q.x = 9;
console.log(p.x, q.y);
