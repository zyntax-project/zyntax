type Read = { readonly x: number };
function identity<T>(x: T): T { return x; }
const r: Read = { x: 1 }; const alias = identity(r); alias.x = 4;
