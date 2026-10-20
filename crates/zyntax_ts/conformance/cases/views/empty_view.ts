interface Empty {}
function alias(p: Empty): Empty { return p; }
const p = { x: 1, y: 2 }; const q: Empty = p; const r = alias(p);
console.log(p === q, p.x);
