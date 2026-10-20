interface Z { z: number }
const p = { a: true, z: 2 }; const q = p as Z;
q.z = 8; console.log(p.z, q.z, p === q);
