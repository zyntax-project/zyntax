class Body {
  constructor(
    public x: number, public y: number, public vx: number, public vy: number,
    public mass: number,
  ) {}
  advance(dt: number): void {
    this.x = this.x + dt * this.vx;
    this.y = this.y + dt * this.vy;
  }
  energy(): number { return 0.5 * this.mass * (this.vx * this.vx + this.vy * this.vy); }
}
function step(bodies: Body[], dt: number): void {
  for (let i = 0; i < bodies.length; i++) {
    const bi = bodies[i];
    for (let j = i + 1; j < bodies.length; j++) {
      const bj = bodies[j];
      const dx = bi.x - bj.x;
      const dy = bi.y - bj.y;
      const d2 = dx * dx + dy * dy + 0.01;
      const mag = dt / (d2 * Math.sqrt(d2));
      bi.vx = bi.vx - dx * bj.mass * mag;
      bi.vy = bi.vy - dy * bj.mass * mag;
      bj.vx = bj.vx + dx * bi.mass * mag;
      bj.vy = bj.vy + dy * bi.mass * mag;
    }
  }
  for (let i = 0; i < bodies.length; i++) { bodies[i].advance(dt); }
}
const bodies: Body[] = [new Body(0, 0, 0, 0, 10), new Body(1, 0, 0, 1, 1), new Body(0, 2, 1, 0, 1)];
for (let s = 0; s < 1000; s++) { step(bodies, 0.01); }
let e = 0;
for (let i = 0; i < bodies.length; i++) { e += bodies[i].energy(); }
console.log(e);
