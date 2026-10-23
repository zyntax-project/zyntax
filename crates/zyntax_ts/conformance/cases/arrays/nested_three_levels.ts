const c: number[][][] = [];
for (let i = 0; i < 3; i++) {
  const p: number[][] = [];
  for (let j = 0; j < 3; j++) { const r: number[] = []; for (let k = 0; k < 3; k++) r.push(i * 9 + j * 3 + k); p.push(r); }
  c.push(p);
}
console.log(c[2][1][0], c[1].length, c[0][2][2]);
