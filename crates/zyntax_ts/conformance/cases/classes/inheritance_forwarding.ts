class Point {
  constructor(public x: number, public y: number = 5) {}
  sum(): number {
    return this.x + this.y;
  }
}
class Named extends Point {
  label = 7;
  sum(): number {
    return this.x + this.y + this.label;
  }
}
class Deeper extends Named {}
function total(p: Point): number {
  return p.sum();
}
const a = new Named(1, 2);
const b = new Named(3);
const c = new Deeper(10, 20);
console.log(total(a), total(b), total(c), c.label);
