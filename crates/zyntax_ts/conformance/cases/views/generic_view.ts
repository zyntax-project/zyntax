interface X { x: number }
function identity<T>(p: T): T { return p; }
const p = { x: 1, y: 2 }; const q: X = p;
const r = identity(q); const s = identity<X>(p);
r.x += 2; s.x += 4; console.log(p.x, q.x, r.x, s.x);
