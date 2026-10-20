type Read = { readonly x: number };
type Write = { x: number };
const r: Read = { x: 1 };
const w: Write = r!;
w.x = 2;
