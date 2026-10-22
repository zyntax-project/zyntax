class Vec {
  x: number;
  y: number;
  scale = 2;
  constructor(x: number, y: number) { this.x = x; this.y = y; }
  dot(o: Vec): number { return this.x * o.x + this.y * o.y; }
  add(o: Vec): Vec { return new Vec(this.x + o.x, this.y + o.y); }
  scaled(): Vec { return new Vec(this.x * this.scale, this.y * this.scale); }
  length2(): number { return this.dot(this); }
  static origin(): Vec { return new Vec(0, 0); }
}
class Counter {
  private n = 0;
  constructor(public readonly step: number, public label: boolean) {}
  tick(): number { this.n = this.n + this.step; return this.n; }
}
const a = new Vec(1, 2);
const b = new Vec(3, 4);
const c = a.add(b).scaled();
console.log(a.dot(b), c.x, c.y, c.length2(), Vec.origin().x);
const k = new Counter(5, true);
k.tick(); k.tick();
console.log(k.tick(), k.step, k.label);
