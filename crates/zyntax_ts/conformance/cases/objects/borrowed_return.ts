interface Point { x: number; } function identity(p: Point): Point { return p; } const p: Point = { x: 1 }; const q: Point = identity(p); q.x = 8; console.log(p.x, p === q);
