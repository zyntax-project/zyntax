interface X { x: number; } interface XY { x: number; y: number; } const p: XY = { x: 1, y: 2 }; const q: X = p; console.log(q.x);
