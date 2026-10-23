function norm(p: { x: number; y: number }): number { return Math.sqrt(p.x * p.x + p.y * p.y); }
function make(x: number): { x: number; y: number } { return { x: x, y: x * 2 }; }
const q: { readonly x: number; y: number } = make(3);
let flag: { on: boolean } = { on: true };
flag.on = !flag.on;
console.log(norm({ x: 3, y: 4 }), norm(make(1)), q.x, q.y, flag.on);
