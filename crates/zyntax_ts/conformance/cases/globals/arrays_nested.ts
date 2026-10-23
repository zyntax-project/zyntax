const grid: number[][] = [];
function row(n: number): void { const r: number[] = []; for (let i = 0; i < n; i++) r.push(i); grid.push(r); }
function total(): number { let t = 0; for (const r of grid) for (const x of r) t += x; return t; }
row(3); row(4);
grid[0].push(10);
console.log(grid.length, grid[0].length, total());
