type Read = { readonly x: number }; type Write = { x: number };
function identity<T>(x: T): T { return x; }
const r: Read = { x: 1 }; const w = identity<Write>(r); w.x = 5; console.log(r.x);
