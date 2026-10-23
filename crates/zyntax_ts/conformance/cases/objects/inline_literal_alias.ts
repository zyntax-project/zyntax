type Point = { x: number; y: number };
function scale(p: { x: number; y: number }, k: number): Point { return { x: p.x * k, y: p.y * k }; }
const a: Point = { x: 1, y: 2 };
const b = scale(a, 3);
console.log(b.x, b.y);
