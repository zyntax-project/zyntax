type Read = { readonly x: number };
type Write = { x: number };
function alias(p: Read) { return p; }
const r: Read = { x: 1 };
const w: Write = alias(r);
w.x = 3;
console.log(r.x);
