interface Point { x: number; } function add(a: Point, b: Point): number { a.x += 1; return a.x + b.x; } const p: Point = { x: 2 }; console.log(add(p, p), p.x);
