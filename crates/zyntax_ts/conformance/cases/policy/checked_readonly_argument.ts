type R = { readonly x: number }; type W = { x: number }; function change(w: W): void { w.x = 2; } const r: R = { x: 1 }; change(r);
