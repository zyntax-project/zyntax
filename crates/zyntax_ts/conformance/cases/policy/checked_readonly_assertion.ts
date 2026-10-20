type R = { readonly x: number }; type W = { x: number }; const r: R = { x: 1 }; const w = r as W;
