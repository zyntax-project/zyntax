interface Point { x: number; y: number; } function echo(x: number): number { console.log(x); return x; } const p: Point = { y: echo(2), x: echo(1) }; console.log(p.x, p.y);
