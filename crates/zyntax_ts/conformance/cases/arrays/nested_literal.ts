const m: number[][] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
let tr = 0;
for (let i = 0; i < 3; i++) tr += m[i][i];
console.log(tr, m.length, m[2].length);
