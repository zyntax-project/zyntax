class Vec { x: number; y: number; constructor(x: number = 0, y: number = x) { this.x = x; this.y = y; }
  scaled(k: number = 2, offset?: number): Vec { return new Vec(this.x * k + (offset === undefined ? 0 : offset), this.y * k); } }
function total(...xs: number[]): number { let t = 0; for (const x of xs) t += x; return t; }
function pick(first: boolean, ...flags: boolean[]): number { return flags.length + (first ? 1 : 0); }
const a = new Vec();
const b = new Vec(3);
const c = new Vec(1, 2).scaled();
const d = c.scaled(10, 1);
console.log(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, total(), total(1, 2, 3), pick(true), pick(false, true, false));
