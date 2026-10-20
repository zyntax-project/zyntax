function identity<T>(x: T): T { return x; }
const p = { x: 1, y: 2 }; const q = identity(p); q.x = 9;
console.log(p.x, q.x, p === q);
