type Point = { x: number }; function value(flag: boolean): number { const p: Point = { x: 4 }; if (flag) return p.x; return p.x + 1; } console.log(value(true), value(false));
