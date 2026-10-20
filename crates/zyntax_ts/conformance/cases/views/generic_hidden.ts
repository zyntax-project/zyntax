interface X { x: number }
function identity<T>(p: T): T { return p; }
const p = { x: 1, y: 2 }; const q: X = p; const r = identity(q); console.log(r.y);
