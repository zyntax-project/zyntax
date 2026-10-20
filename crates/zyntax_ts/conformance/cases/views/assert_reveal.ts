interface X { x: number } interface XY { x: number; y: number }
const p: XY = { x: 1, y: 2 }; const q: X = p; const r = q as XY;
r.y = 7; console.log(p.x, p.y, r.y, p === r);
