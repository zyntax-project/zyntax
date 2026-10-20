interface Z { z: number }
function make(n: number) { const p = { a: true, z: n }; const q: Z = p; return q; }
const q = make(4); q.z += 1; console.log(q.z);
