interface X { x: number }
function alias(p: X): X { return p; }
const p = { x: 1, y: 2 }; const q = alias(p); q.x = 9;
console.log(p.x, p.y, q.x);
