interface X { x: number } interface XY { x: number; y: number }
function wide(p: XY): number { return p.y; }
const p: XY = { x: 1, y: 2 }; const q: X = p; console.log(wide(q));
