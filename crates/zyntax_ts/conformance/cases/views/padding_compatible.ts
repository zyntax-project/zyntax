interface BZ { b: boolean; z: number }
function bump(p: BZ): BZ { p.z += 1; return p; }
const p = { b: true, c: false, z: 2 }; const q = bump(p);
q.b = false; console.log(p.b, p.c, p.z, q.z);
