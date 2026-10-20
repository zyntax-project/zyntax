interface Z { z: number }
function make() { const p = { a: true, z: 1 }; const q: Z = p; return q; }
const q = make(); console.log(q.a);
