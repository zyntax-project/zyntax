function make(n: number, seed: number): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < n; i++) {
    const r: number[] = [];
    for (let j = 0; j < n; j++) r.push((i * 31 + j * 17 + seed) % 10);
    m.push(r);
  }
  return m;
}
function trace(a: number[][], n: number): number {
  let t = 0;
  for (let i = 0; i < n; i++) t += a[i][i];
  return t;
}
function grow(a: number[][], n: number): number {
  const row = a[0];
  for (let i = 0; i < n; i++) row.push(i);
  a[1].push(row.length);
  return a[0].length + a[1].length;
}
let total = 0;
for (let it = 0; it < 300; it++) {
  const m = make(20, it);
  total += trace(m, 20);
  total += grow(m, 5);
  const alias = m[3];
  alias[2] = it;
  total += m[3][2];
}
console.log(total);
