interface Point { x: number; y: number; } function sum(p: Point): number { return p.x * 10 + p.y; } console.log(sum({ y: 2, x: 1 }));
