class Point { x: number; y: number; constructor(x: number, y: number) { this.x = x; this.y = y; } }
const points: Point[] = [];
function add(x: number, y: number): void { points.push(new Point(x, y)); }
function far(): number { let best = 0; for (const p of points) { const d = p.x * p.x + p.y * p.y; if (d > best) best = d; } return best; }
add(1, 2); add(3, 4); add(-5, 0);
const first = points[0];
first.x = 10;
console.log(points.length, far(), first.x);
