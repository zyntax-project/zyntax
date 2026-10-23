const grid: number[][] = [[1, 2], [3, 4]];
const row = grid[0];
row[1] = 9;
row.push(7);
grid[1].push(8);
console.log(grid[0][1], grid[0].length, grid[1].length, grid[1][2], row.length);
const same = grid[0] === row;
console.log(same);
