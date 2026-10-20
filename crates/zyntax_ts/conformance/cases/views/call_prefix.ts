interface X { x: number }
function bump(p: X): void { p.x += 2; }
const p = { x: 1, z: 4 }; bump(p); console.log(p.x, p.z);
