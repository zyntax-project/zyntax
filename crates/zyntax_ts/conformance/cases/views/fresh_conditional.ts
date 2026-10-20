interface X { x: number }
const flag = true; const q: X = flag ? { x: 1, y: 2 } : { x: 2, y: 3 };
