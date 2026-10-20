type R = { readonly x: number }; type W = { x: number }; function mutable(r: R): W { return r; }
