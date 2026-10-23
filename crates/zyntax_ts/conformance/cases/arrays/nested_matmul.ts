function identity(n: number): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < n; i++) {
    const r: number[] = [];
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    m.push(r);
  }
  return m;
}
function mul(a: number[][], b: number[][], n: number): number[][] {
  const c: number[][] = [];
  for (let i = 0; i < n; i++) {
    const r: number[] = [];
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += a[i][k] * b[k][j];
      r.push(s);
    }
    c.push(r);
  }
  return c;
}
const n = 5;
const a = identity(n);
a[1][2] = 3; a[3][0] = 2;
const b = mul(a, a, n);
let t = 0;
for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) t += b[i][j] * (i + 1) * (j + 2);
console.log(t);
