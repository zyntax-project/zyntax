type R = { readonly x: number }; type W = { x: number }; const r: R = { x: 1 }; let w: W = { x: 2 }; w = r;
