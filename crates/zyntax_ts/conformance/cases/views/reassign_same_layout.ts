interface X { x: number }
const a = { x: 1, y: 2 }; const b = { x: 3, y: 4 };
let q: X = a; q = b; q.x = 9; console.log(a.x, b.x, q.x, q === b);
