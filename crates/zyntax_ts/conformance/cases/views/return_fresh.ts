interface X { x: number }
function make(n: number): X { const p = { x: n, y: 2 }; return p; }
const p = make(5); p.x += 1; console.log(p.x);
