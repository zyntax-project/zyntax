interface Point { x: number; y: number }
interface Point3 extends Point { z: number }
interface Labeled { label: string }
interface Pin extends Point, Labeled { pinned: boolean }
type Size = { w: number; h: number };
interface Box extends Size { depth: number }
function norm(p: Point): number { return Math.sqrt(p.x * p.x + p.y * p.y); }
function tag(p: Pin): string { return p.label + (p.pinned ? "!" : ""); }
function volume(b: Box): number { return b.w * b.h * b.depth; }
const p3: Point3 = { x: 3, y: 4, z: 12 };
const pin: Pin = { x: 1, y: 0, label: "home", pinned: true };
class Spot implements Point, Labeled { x = 0; y = 0; label = "spot"; constructor(x: number) { this.x = x; } }
const spot = new Spot(6);
console.log(norm(p3), p3.z, tag(pin), pin.x, volume({ w: 2, h: 3, depth: 4 }), spot.label, spot.x);
