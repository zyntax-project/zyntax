interface Point { x: number; y: number; } const p: Point = { x: 1, y: 2 }; const alias: Point = p; alias.x = 9; console.log(p.x, alias.y, p === alias);
