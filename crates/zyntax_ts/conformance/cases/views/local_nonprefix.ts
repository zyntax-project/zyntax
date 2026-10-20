interface Z { z: number }
const p = { a: true, x: 2, z: 3 }; const q: Z = p;
q.z += 4; console.log(p.a, p.x, p.z, q.z, p === q);
