interface X { x: number }
const p = { x: 1, y: 2 }; const q: X = p; const r = (q);
r.x = 8; console.log(p.x, q.x, r.x, p === r);
