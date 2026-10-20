type R = { readonly x: number }; type W = { x: number }; const r: R = { x: 1 }; const w: W = r; w.x = 2; console.log(r.x);
