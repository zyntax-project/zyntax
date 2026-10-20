interface Read { readonly x: number } interface Write { x: number }
const p = { x: 1, y: 2 }; const q: Read = p; const r: Write = q;
r.x = 7; console.log(p.x, q.x, r.x);
