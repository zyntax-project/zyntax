function a(i: number, j: number): number { return 1 / ((i + j) * (i + j + 1) / 2 + i + 1); }
function mulAv(n: number, v: number[], av: number[]): void {
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) { s += a(i, j) * v[j]; } av[i] = s; }
}
function mulAtv(n: number, v: number[], atv: number[]): void {
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) { s += a(j, i) * v[j]; } atv[i] = s; }
}
function run(n: number): number {
  const u: number[] = []; const v: number[] = []; const t: number[] = [];
  for (let i = 0; i < n; i++) { u.push(1); v.push(0); t.push(0); }
  for (let k = 0; k < 10; k++) { mulAv(n, u, t); mulAtv(n, t, v); mulAv(n, v, t); mulAtv(n, t, u); }
  let vbv = 0, vv = 0;
  for (let i = 0; i < n; i++) { vbv += u[i] * v[i]; vv += v[i] * v[i]; }
  return Math.sqrt(vbv / vv);
}
console.log(run(20));
