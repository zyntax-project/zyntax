interface Point { x: number; y: number; } function make(x: number): Point { return { y: 2, x: x }; } const p: Point = make(3); console.log(p.x, p.y);
